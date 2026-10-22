//! Searches requested by other apps (launchers such as Tuna or Raycast).
//!
//! The window shows every match and lets the user sort them, but a launcher only
//! shows the first few dozen. So external results are ranked for that case:
//! 1. noise comes last, however well its name matches. Noise the user may have
//!    edited (dotfiles, ~/Library, system settings) is exempt when it changed in the
//!    last 7 days; generated noise (dependencies, build output, caches, app
//!    internals, the OS itself) never is,
//! 2. then by how well the file name matches (exact, then prefix, then contains),
//! 3. then by where the file lives (home folder before the rest of the disk),
//! 4. then by modification time (newest first).

use anyhow::Result;
use crossbeam_channel::Sender;
use search_cache::{SearchCache, SearchOptions, SearchResultNode, SlabIndex};
use search_cancel::CancellationToken;
use std::{
    collections::HashMap,
    hash::{BuildHasherDefault, Hasher},
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

/// How many of the best candidates get their metadata loaded for the final ordering.
const CANDIDATE_POOL: usize = 2000;

/// Extra candidates from noise the user may have edited, so a recently edited one can
/// still make the list when plenty of ordinary files match.
const EDITABLE_NOISE_POOL: usize = 500;

/// Editable noise changed within this many seconds ranks like an ordinary file.
const RECENT_EDIT_WINDOW_SECS: u32 = 7 * 24 * 60 * 60;

/// Top-level folders that hold the operating system itself.
const OS_ROOT_DIRS: &[&str] = &["System", "usr", "bin", "sbin", "cores", "dev"];

/// Top-level folders that hold system-wide settings and data the user may edit.
const SYSTEM_DATA_ROOT_DIRS: &[&str] = &["Library", "private", "opt", "etc", "var", "tmp"];

/// Folders whose contents are cached, downloaded or built rather than made by the user.
const GENERATED_DIR_NAMES: &[&str] = &[
    "node_modules",
    "__pycache__",
    "DerivedData",
    "Caches",
    "target",
    "build",
    "dist",
    ".git",
    ".cache",
    ".npm",
    ".cargo",
    ".rustup",
    ".nvm",
    ".pnpm-store",
    ".gradle",
    ".m2",
    ".venv",
    ".build",
    ".swiftpm",
    ".next",
    ".turbo",
    ".vercel",
    ".expo",
    ".parcel-cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".terraform",
];

/// Hidden files that macOS writes on its own, often several times a day.
const GENERATED_HIDDEN_FILES: &[&str] = &[".DS_Store", ".localized", ".CFUserTextEncoding"];

/// Package folders. The package itself is a useful result, its contents are not.
const PACKAGE_SUFFIXES: &[&str] = &[
    ".app",
    ".framework",
    ".bundle",
    ".plugin",
    ".appex",
    ".kext",
    ".photoslibrary",
];

#[derive(Debug)]
pub struct ExternalSearchJob {
    pub query: String,
    pub case_insensitive: bool,
    pub limit: usize,
    pub cancellation_token: CancellationToken,
    /// `Ok(None)` means the search was cancelled by a newer external search.
    pub result_tx: Sender<Result<Option<ExternalSearchResults>>>,
}

#[derive(Debug)]
pub struct ExternalSearchResults {
    /// Number of matches before ranking and truncation.
    pub total: usize,
    pub nodes: Vec<SearchResultNode>,
}

pub fn handle_external_search(cache: &mut SearchCache, job: ExternalSearchJob) {
    let ExternalSearchJob {
        query,
        case_insensitive,
        limit,
        cancellation_token,
        result_tx,
    } = job;
    let home = std::env::var_os("HOME");
    let result = run_external_search(
        cache,
        &query,
        case_insensitive,
        limit,
        home.as_deref().map(Path::new),
        cancellation_token,
    );
    // The requester may have given up already; nothing to do then.
    let _ = result_tx.send(result);
}

pub(crate) fn run_external_search(
    cache: &mut SearchCache,
    query: &str,
    case_insensitive: bool,
    limit: usize,
    home: Option<&Path>,
    cancellation_token: CancellationToken,
) -> Result<Option<ExternalSearchResults>> {
    if query.trim().is_empty() {
        return Ok(Some(ExternalSearchResults {
            total: 0,
            nodes: Vec::new(),
        }));
    }

    let outcome = cache.search_with_options(
        query,
        SearchOptions { case_insensitive },
        cancellation_token,
    )?;
    let Some(matches) = outcome.nodes else {
        return Ok(None);
    };
    let total = matches.len();

    let name_query = NameQuery::new(query, case_insensitive);
    let mut places = PlaceClassifier::new(home.and_then(|home| cache.node_index_for_path(home)));
    let mut candidates = Vec::with_capacity(matches.len());
    for (counter, index) in matches.into_iter().enumerate() {
        if cancellation_token.is_cancelled_sparse(counter).is_none() {
            return Ok(None);
        }
        let Some(name) = cache.node_name(index) else {
            continue;
        };
        let name_rank = name_query
            .as_ref()
            .map_or(NameRank::Exact, |name_query| name_query.rank(name));
        let place = places.place_of(cache, index, name);
        candidates.push((CandidateKey { name_rank, place }, index));
    }

    // Loading metadata touches the disk, so only do it for the best candidates. Editable
    // noise gets its own share: its modification time decides whether it sinks.
    let (mut editable_noise, mut candidates): (Vec<_>, Vec<_>) = candidates
        .into_iter()
        .partition(|(key, _)| key.place.is_editable_noise());
    keep_best(&mut candidates, CANDIDATE_POOL, |key| key.pool_order());
    keep_best(&mut editable_noise, EDITABLE_NOISE_POOL, |key| {
        (key.name_rank, key.place.depth)
    });
    candidates.append(&mut editable_noise);
    let indices: Vec<SlabIndex> = candidates.iter().map(|(_, index)| *index).collect();
    let nodes = cache.expand_file_nodes(&indices);

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| u32::try_from(elapsed.as_secs()).unwrap_or(u32::MAX))
        .unwrap_or_default();
    let mut ranked: Vec<(RankKey, SearchResultNode)> = candidates
        .into_iter()
        .zip(nodes)
        .map(|((key, _), node)| (key.final_rank(modified_at(&node), now), node))
        .collect();
    ranked.sort_by(|(a_key, a_node), (b_key, b_node)| {
        a_key
            .cmp(b_key)
            .then_with(|| modified_at(b_node).cmp(&modified_at(a_node)))
            .then_with(|| a_node.path.cmp(&b_node.path))
    });
    ranked.truncate(limit);

    Ok(Some(ExternalSearchResults {
        total,
        nodes: ranked.into_iter().map(|(_, node)| node).collect(),
    }))
}

/// Keeps the `count` candidates with the smallest `order`, in no particular order.
fn keep_best<K: Ord>(
    candidates: &mut Vec<(CandidateKey, SlabIndex)>,
    count: usize,
    order: impl Fn(&CandidateKey) -> K,
) {
    if candidates.len() > count {
        candidates.select_nth_unstable_by_key(count - 1, |(key, _)| order(key));
        candidates.truncate(count);
    }
}

fn modified_at(node: &SearchResultNode) -> u32 {
    node.metadata
        .as_ref()
        .and_then(|metadata| metadata.mtime())
        .map(|mtime| mtime.get())
        .unwrap_or_default()
}

#[derive(Debug, Clone, Copy)]
struct CandidateKey {
    name_rank: NameRank,
    place: Place,
}

/// Final order before modification time: noise last, then name match, then place.
/// Within noise, what the user may have edited comes before what programs generate.
type RankKey = (bool, NameRank, PlaceTier, Option<NoiseKind>);

impl CandidateKey {
    fn final_rank(&self, modified: u32, now: u32) -> RankKey {
        let tier = self.place.tier_given_edit(modified, now);
        let is_noise = tier == PlaceTier::Noise;
        let noise = if is_noise { self.place.noise } else { None };
        (is_noise, self.name_rank, tier, noise)
    }

    /// Order used to pick the candidate pool, before modification times are known.
    /// Shallower files win ties, since they are more likely to be what the user meant.
    fn pool_order(&self) -> (bool, NameRank, PlaceTier, u16) {
        (
            self.place.tier == PlaceTier::Noise,
            self.name_rank,
            self.place.tier,
            self.place.depth,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum NameRank {
    /// The name, or the name without its extension, equals the query.
    Exact,
    /// The name starts with the first word of the query.
    Prefix,
    Contains,
}

/// Plain words of a query, without filters, operators, wildcards or path parts.
/// `ext:pdf "annual report" !draft` gives `["annual report"]`.
pub(crate) fn name_terms(query: &str, case_insensitive: bool) -> Vec<String> {
    let mut terms = Vec::new();
    let mut chars = query.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }
        let mut token = String::new();
        let quoted = c == '"';
        if quoted {
            chars.next();
            for c in chars.by_ref() {
                if c == '"' {
                    break;
                }
                token.push(c);
            }
        } else {
            while let Some(&c) = chars.peek() {
                if c.is_whitespace() {
                    break;
                }
                token.push(c);
                chars.next();
            }
        }

        let token = token.trim_matches('/');
        let is_operator = !quoted && matches!(token, "AND" | "OR" | "NOT");
        let has_syntax = token.starts_with('!')
            || token
                .chars()
                .any(|c| matches!(c, ':' | '/' | '*' | '?' | '|' | ';' | '<' | '>'));
        if token.is_empty() || is_operator || (!quoted && has_syntax) {
            continue;
        }
        terms.push(if case_insensitive {
            token.to_lowercase()
        } else {
            token.to_string()
        });
    }
    terms
}

/// The plain words of a query, prepared once and compared against every match.
pub(crate) struct NameQuery {
    /// All words joined by spaces, for exact matches.
    phrase: String,
    /// The first word, for prefix matches.
    first: String,
    case_insensitive: bool,
}

impl NameQuery {
    /// Returns `None` when the query has no plain words, e.g. `ext:pdf size:>1MB`.
    pub(crate) fn new(query: &str, case_insensitive: bool) -> Option<Self> {
        let terms = name_terms(query, case_insensitive);
        let first = terms.first()?.clone();
        Some(Self {
            phrase: terms.join(" "),
            first,
            case_insensitive,
        })
    }

    pub(crate) fn rank(&self, name: &str) -> NameRank {
        // People type "zshrc" for ".zshrc", so a leading dot does not count.
        let name = match name.strip_prefix('.') {
            Some(rest) if !rest.is_empty() => rest,
            _ => name,
        };
        // The phrase starts with the first word, so an exact match is always a prefix
        // match too. Checking the prefix first lets most names stop here.
        if !self.starts_with(name, &self.first) {
            return NameRank::Contains;
        }
        let stem = match name.rfind('.') {
            Some(dot) if dot > 0 => &name[..dot],
            _ => name,
        };
        if self.equals(name, &self.phrase) || self.equals(stem, &self.phrase) {
            NameRank::Exact
        } else {
            NameRank::Prefix
        }
    }

    // Both comparisons run for every match, so they fold case without allocating.
    fn equals(&self, name: &str, term: &str) -> bool {
        if self.case_insensitive {
            name.chars().flat_map(char::to_lowercase).eq(term.chars())
        } else {
            name == term
        }
    }

    fn starts_with(&self, name: &str, prefix: &str) -> bool {
        if !self.case_insensitive {
            return name.starts_with(prefix);
        }
        // Most names start with ASCII, where comparing bytes is much cheaper.
        if prefix.is_ascii()
            && let Some(head) = name.as_bytes().get(..prefix.len())
            && head.is_ascii()
        {
            return head.eq_ignore_ascii_case(prefix.as_bytes());
        }
        let mut folded = name.chars().flat_map(char::to_lowercase);
        prefix.chars().all(|c| folded.next() == Some(c))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum PlaceTier {
    /// Inside the user's home folder.
    Home,
    /// Elsewhere on disk, such as apps in /Applications or external volumes.
    Elsewhere,
    /// System files, app internals, caches, dependencies, build output and hidden files.
    /// Ranked last, whatever the name match.
    Noise,
}

/// Why a file counts as noise. Ordered so the stronger reason wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum NoiseKind {
    /// Dotfiles, ~/Library and system settings: the user may have edited these.
    Editable,
    /// Dependencies, build output, caches, app internals and the OS itself.
    Generated,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Place {
    pub(crate) tier: PlaceTier,
    in_home: bool,
    noise: Option<NoiseKind>,
    depth: u16,
}

impl Place {
    fn is_editable_noise(&self) -> bool {
        self.noise == Some(NoiseKind::Editable)
    }

    /// Editable noise changed recently ranks like an ordinary file in the same place.
    fn tier_given_edit(&self, modified: u32, now: u32) -> PlaceTier {
        if self.is_editable_noise() && modified.saturating_add(RECENT_EDIT_WINDOW_SECS) >= now {
            if self.in_home {
                PlaceTier::Home
            } else {
                PlaceTier::Elsewhere
            }
        } else {
            self.tier
        }
    }
}

/// What a folder means for the files inside it.
#[derive(Debug, Clone, Copy)]
struct FolderContext {
    in_home: bool,
    noise: Option<NoiseKind>,
    depth: u16,
}

/// Hashes slab indices with a single multiply. Short queries can touch hundreds of
/// thousands of folders, and the default hasher is built for untrusted keys.
#[derive(Default)]
struct SlabIndexHasher(u64);

impl Hasher for SlabIndexHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0.rotate_left(8) ^ u64::from(byte)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }

    fn write_u32(&mut self, n: u32) {
        self.0 = u64::from(n).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

/// Classifies matches by location, remembering each folder so siblings are cheap.
struct PlaceClassifier {
    home: Option<SlabIndex>,
    folders: HashMap<SlabIndex, FolderContext, BuildHasherDefault<SlabIndexHasher>>,
    /// Matches arrive in path order, so consecutive ones usually share a folder.
    last_folder: Option<(SlabIndex, FolderContext)>,
}

impl PlaceClassifier {
    fn new(home: Option<SlabIndex>) -> Self {
        Self {
            home,
            folders: HashMap::default(),
            last_folder: None,
        }
    }

    fn place_of(&mut self, cache: &SearchCache, index: SlabIndex, name: &str) -> Place {
        let Some(parent) = cache.node_parent(index) else {
            return Place {
                tier: PlaceTier::Elsewhere,
                in_home: false,
                noise: None,
                depth: 0,
            };
        };
        let context = match self.last_folder {
            Some((folder, context)) if folder == parent => context,
            _ => {
                let context = self.folder_context(cache, parent);
                self.last_folder = Some((parent, context));
                context
            }
        };
        let noise = context.noise.max(file_noise(name));
        let tier = if noise.is_some() {
            PlaceTier::Noise
        } else if context.in_home {
            PlaceTier::Home
        } else {
            PlaceTier::Elsewhere
        };
        Place {
            tier,
            in_home: context.in_home,
            noise,
            depth: context.depth,
        }
    }

    fn folder_context(&mut self, cache: &SearchCache, folder: SlabIndex) -> FolderContext {
        if let Some(context) = self.folders.get(&folder) {
            return *context;
        }

        // Walk up to the root or to a folder we already know, then fill in downwards.
        let mut chain = vec![folder];
        let mut known = None;
        let mut current = folder;
        while let Some(parent) = cache.node_parent(current) {
            if let Some(context) = self.folders.get(&parent) {
                known = Some(*context);
                break;
            }
            chain.push(parent);
            current = parent;
        }

        let mut context = known;
        for &folder in chain.iter().rev() {
            let next = match context {
                None => FolderContext {
                    in_home: Some(folder) == self.home,
                    noise: None,
                    depth: 0,
                },
                Some(parent_context) => {
                    let name = cache.node_name(folder).unwrap_or_default();
                    let parent = cache.node_parent(folder);
                    let parent_is_root = parent.is_some_and(|p| cache.node_parent(p).is_none());
                    let parent_is_home = parent.is_some() && parent == self.home;
                    FolderContext {
                        in_home: parent_context.in_home || Some(folder) == self.home,
                        noise: parent_context.noise.max(folder_noise(
                            name,
                            parent_is_root,
                            parent_is_home,
                        )),
                        depth: parent_context.depth.saturating_add(1),
                    }
                }
            };
            self.folders.insert(folder, next);
            context = Some(next);
        }
        context.expect("chain always holds the starting folder")
    }
}

/// Whether a file is noise on its own, whatever folder it is in.
pub(crate) fn file_noise(name: &str) -> Option<NoiseKind> {
    if GENERATED_HIDDEN_FILES.contains(&name) || name.starts_with("._") {
        // `._name` files hold Finder metadata copied to non-Mac disks.
        Some(NoiseKind::Generated)
    } else if name.starts_with('.') {
        Some(NoiseKind::Editable)
    } else {
        None
    }
}

/// Whether files inside this folder are noise, and why.
pub(crate) fn folder_noise(
    name: &str,
    parent_is_root: bool,
    parent_is_home: bool,
) -> Option<NoiseKind> {
    if GENERATED_DIR_NAMES.contains(&name)
        || PACKAGE_SUFFIXES.iter().any(|suffix| name.ends_with(suffix))
        || (parent_is_root && OS_ROOT_DIRS.contains(&name))
    {
        Some(NoiseKind::Generated)
    } else if name.starts_with('.')
        || (parent_is_root && SYSTEM_DATA_ROOT_DIRS.contains(&name))
        || (parent_is_home && name == "Library")
    {
        Some(NoiseKind::Editable)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        time::{Duration, SystemTime},
    };
    use tempfile::TempDir;

    /// Temp folders normally live under /private and start with a dot, and the build
    /// folder is called `target`; the ranking treats all of those as noise. So create
    /// them next to the crate manifest instead. They are removed when the test ends.
    fn temp_root() -> (TempDir, std::path::PathBuf) {
        let temp = tempfile::Builder::new()
            .prefix("external-search-")
            .tempdir_in(env!("CARGO_MANIFEST_DIR"))
            .unwrap();
        let root = temp.path().canonicalize().unwrap();
        (temp, root)
    }

    #[test]
    fn name_terms_skip_filters_operators_and_wildcards() {
        assert_eq!(
            name_terms(r#"ext:pdf "Annual Report" !draft *.psd OR /notes/"#, true),
            vec!["annual report".to_string(), "notes".to_string()]
        );
        assert_eq!(name_terms("Report", false), vec!["Report".to_string()]);
        assert!(name_terms("size:>1MB a|b", true).is_empty());
    }

    #[test]
    fn name_rank_prefers_exact_then_prefix() {
        let query = NameQuery::new("report", true).unwrap();
        assert_eq!(query.rank("Report.pdf"), NameRank::Exact);
        assert_eq!(query.rank("report"), NameRank::Exact);
        assert_eq!(query.rank("report-2024.pdf"), NameRank::Prefix);
        assert_eq!(query.rank("REPORTS"), NameRank::Prefix);
        assert_eq!(query.rank("old report.pdf"), NameRank::Contains);
        assert_eq!(query.rank("rep"), NameRank::Contains);

        let case_sensitive = NameQuery::new("report", false).unwrap();
        assert_eq!(case_sensitive.rank("Report.pdf"), NameRank::Contains);

        let phrase = NameQuery::new(r#""Annual Report""#, true).unwrap();
        assert_eq!(phrase.rank("annual report.key"), NameRank::Exact);
        assert_eq!(phrase.rank("Annual Report 2024.key"), NameRank::Prefix);

        let accented = NameQuery::new("CAFÉ", true).unwrap();
        assert_eq!(accented.rank("Café.txt"), NameRank::Exact);
        assert_eq!(accented.rank("Cafés"), NameRank::Prefix);
        let chinese = NameQuery::new("报告", true).unwrap();
        assert_eq!(chinese.rank("报告2024.pdf"), NameRank::Prefix);
        assert_eq!(chinese.rank("年度报告.pdf"), NameRank::Contains);

        // A leading dot does not count: "zshrc" finds ".zshrc".
        let dotfile = NameQuery::new("zshrc", true).unwrap();
        assert_eq!(dotfile.rank(".zshrc"), NameRank::Exact);
        assert_eq!(dotfile.rank(".zshrc.bak"), NameRank::Exact);
        assert_eq!(dotfile.rank("zshrc_Apple_Terminal"), NameRank::Prefix);

        assert!(NameQuery::new("ext:pdf size:>1MB", true).is_none());
    }

    #[test]
    fn folder_noise_tells_editable_from_generated() {
        use NoiseKind::{Editable, Generated};
        assert_eq!(folder_noise("System", true, false), Some(Generated));
        assert_eq!(folder_noise("System", false, false), None);
        assert_eq!(folder_noise("Library", true, false), Some(Editable));
        assert_eq!(folder_noise("Library", false, true), Some(Editable));
        assert_eq!(folder_noise("Library", false, false), None);
        assert_eq!(folder_noise(".config", false, true), Some(Editable));
        assert_eq!(folder_noise(".git", false, false), Some(Generated));
        assert_eq!(folder_noise("node_modules", false, false), Some(Generated));
        assert_eq!(folder_noise("target", false, false), Some(Generated));
        assert_eq!(folder_noise("Safari.app", false, false), Some(Generated));
        assert_eq!(folder_noise("Documents", false, true), None);
        assert_eq!(folder_noise(".build", false, false), Some(Generated));

        assert_eq!(file_noise(".zshrc"), Some(Editable));
        assert_eq!(file_noise(".DS_Store"), Some(Generated));
        assert_eq!(file_noise("._report.pdf"), Some(Generated));
        assert_eq!(file_noise("report.pdf"), None);
    }

    #[test]
    fn recently_edited_noise_ranks_like_an_ordinary_file() {
        let (_temp, root) = temp_root();
        let now = SystemTime::now();
        let long_ago = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        touch_at(&root, "home/Documents/report.pdf", long_ago);
        touch_at(&root, "home/Documents/report-draft.pdf", long_ago);
        // Editable noise: rises when changed in the last 7 days, sinks otherwise.
        touch_at(&root, "home/.config/report.toml", now);
        touch_at(
            &root,
            "home/Library/Application Support/App/report.json",
            long_ago,
        );
        // Generated noise: sinks however recent.
        touch_at(&root, "home/project/target/release/report", now);
        touch_at(&root, "home/.config/app/node_modules/report.js", now);

        let mut cache = SearchCache::walk_fs(&root);
        let results = run_external_search(
            &mut cache,
            "report",
            true,
            50,
            Some(&root.join("home")),
            CancellationToken::noop(),
        )
        .unwrap()
        .unwrap();
        let order: Vec<_> = results
            .nodes
            .iter()
            .map(|node| node.path.strip_prefix(&root).unwrap().to_path_buf())
            .collect();
        assert_eq!(
            order,
            [
                // Exact matches in home, newest first: the recently edited dotfile leads.
                "home/.config/report.toml",
                "home/Documents/report.pdf",
                "home/Documents/report-draft.pdf",
                // Noise: editable before generated, then newest first, then by path.
                "home/Library/Application Support/App/report.json",
                "home/.config/app/node_modules/report.js",
                "home/project/target/release/report",
            ]
            .map(std::path::PathBuf::from)
        );
    }

    /// Creates a file with a fixed modification time so ties fall back to path order.
    fn touch(root: &Path, relative: &str) {
        touch_at(
            root,
            relative,
            SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000),
        );
    }

    fn touch_at(root: &Path, relative: &str, modified: SystemTime) {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"x").unwrap();
        fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(modified)
            .unwrap();
    }

    #[test]
    fn ranks_home_files_first_and_noise_last() {
        let (_temp, root) = temp_root();
        touch(&root, "home/Documents/report.pdf");
        touch(&root, "home/Documents/report-draft.pdf");
        touch(&root, "home/Library/Caches/report.pdf");
        touch(&root, "home/project/node_modules/pkg/report.js");
        touch(&root, "home/project/target/release/report");
        touch(&root, "home/.config/report.toml");
        touch(&root, "home/Notes/old report.txt");
        touch(&root, "shared/report.pdf");
        touch(&root, "Tools.app/Contents/report.pdf");

        let mut cache = SearchCache::walk_fs(&root);
        let home = root.join("home");
        let results = run_external_search(
            &mut cache,
            "report",
            true,
            50,
            Some(&home),
            CancellationToken::noop(),
        )
        .unwrap()
        .expect("search should not be cancelled");

        let order: Vec<String> = results
            .nodes
            .iter()
            .map(|node| {
                node.path
                    .strip_prefix(&root)
                    .unwrap()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert_eq!(results.total, order.len());
        assert_eq!(
            order,
            vec![
                // Exact matches: home first, then elsewhere.
                "home/Documents/report.pdf",
                "shared/report.pdf",
                // Prefix matches.
                "home/Documents/report-draft.pdf",
                // Contains matches.
                "home/Notes/old report.txt",
                // Noise last, even when the name is an exact match. What the user may
                // have edited comes before what programs generate.
                "home/.config/report.toml",
                "Tools.app/Contents/report.pdf",
                "home/Library/Caches/report.pdf",
                "home/project/node_modules/pkg/report.js",
                "home/project/target/release/report",
            ]
        );
    }

    #[test]
    fn newer_files_come_first_within_the_same_tier() {
        let (_temp, root) = temp_root();
        let base = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        touch_at(&root, "home/a/plan.md", base);
        touch_at(&root, "home/b/plan.md", base + Duration::from_secs(60));
        let mut cache = SearchCache::walk_fs(&root);
        let results = run_external_search(
            &mut cache,
            "plan",
            true,
            50,
            Some(&root.join("home")),
            CancellationToken::noop(),
        )
        .unwrap()
        .unwrap();
        let first = results.nodes[0].path.strip_prefix(&root).unwrap();
        assert_eq!(first, Path::new("home/b/plan.md"));
    }

    #[test]
    fn respects_limit_and_reports_total() {
        let (_temp, root) = temp_root();
        for i in 0..10 {
            touch(&root, &format!("home/note-{i}.txt"));
        }
        let mut cache = SearchCache::walk_fs(&root);
        let results = run_external_search(
            &mut cache,
            "note",
            true,
            3,
            Some(&root.join("home")),
            CancellationToken::noop(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(results.total, 10);
        assert_eq!(results.nodes.len(), 3);
    }

    #[test]
    fn empty_query_returns_nothing() {
        let (_temp, root) = temp_root();
        touch(&root, "home/a.txt");
        let mut cache = SearchCache::walk_fs(&root);
        let results =
            run_external_search(&mut cache, "  ", true, 50, None, CancellationToken::noop())
                .unwrap()
                .unwrap();
        assert_eq!(results.total, 0);
        assert!(results.nodes.is_empty());
    }
}
