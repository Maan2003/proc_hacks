#![feature(proc_macro_span)]
#![feature(default_free_fn)]

// mod reload;

use std::convert::TryInto;
use std::default::default;
use std::error::Error;
use std::sync::Arc;

use crossbeam_channel::{unbounded, Receiver};
use hir::HirDisplay;
use ide::{AnalysisHost, Change, LineCol, SourceRoot, TextRange};
use ide_db::base_db::CrateGraph;
use ide_db::defs::NameRefClass;
use ide_db::{base_db::SourceDatabaseExt, LineIndexDatabase};
use project_model::{
    BuildDataResult, CargoConfig, ProcMacroClient, ProjectManifest, ProjectWorkspace,
};
use syntax::algo::{self, replace_children};
use syntax::ast::{self, make};
use syntax::{AstNode, SourceFile, SyntaxNode};
use text_edit::TextEditBuilder;
use vfs::file_set::FileSetConfig;
use vfs::VfsPath;
use vfs::{loader::Handle, AbsPath, AbsPathBuf};

#[proc_macro]
pub fn type_of(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let (mut host, vfs, _) = load_workspace_at(
        std::env::var("CARGO_MANIFEST_DIR")
            .unwrap()
            .as_str()
            .try_into()
            .unwrap(),
        &default(),
        &|_| {},
    )
    .unwrap();

    let call_site = proc_macro::Span::call_site();
    let path = call_site.source_file().path().canonicalize().unwrap();
    let file_id = vfs.file_id(&AbsPathBuf::assert(path).into()).unwrap();
    let db = host.raw_database();
    let line_idx = db.line_index(file_id);
    let start = line_idx.offset(dbg!(line_col(call_site.start())));
    let end = line_idx.offset(dbg!(line_col(call_site.end())));
    let range = dbg!(TextRange::new(start, end));

    let sema = hir::Semantics::new(host.raw_database());
    let file = sema.parse(file_id);
    let syn = dbg!(file.syntax());
    let call: ast::MacroCall = algo::find_node_at_range(syn, range).unwrap();

    let ident = call
        .token_tree()
        .unwrap()
        .syntax()
        .children_with_tokens()
        .nth(1)
        .unwrap();
    let var_name = ident.into_token().unwrap();
    let var_name = var_name.text();

    let range = if let Some(ty) = call.syntax().parent().and_then(ast::TypeAlias::cast) {
        ty.syntax().text_range()
    } else {
        call.syntax().text_range()
    };

    let mut edit = TextEditBuilder::default();
    edit.replace(range, var_name.to_string());
    let edit = edit.finish();
    let db = host.raw_database_mut();
    let mut txt = db.file_text(file_id);
    edit.apply(Arc::make_mut(&mut txt));
    db.set_file_text(file_id, txt);

    let sema = hir::Semantics::new(db);
    let file = sema.parse(file_id);
    let syn = dbg!(file.syntax());
    let range = TextRange::at(range.start(), (var_name.len() as u32).into());
    let name_ref: ast::NameRef = dbg!(syntax::algo::find_node_at_range(syn, range).unwrap());
    let ty = match NameRefClass::classify(&sema, &name_ref)
        .unwrap()
        .referenced(db)
    {
        ide_db::defs::Definition::Local(l) => l.ty(db).display(db).to_string(),
        _ => unreachable!(),
    };

    // let name: ast::Path = ast_from_text("fn f() { x; }");
    // let node = dbg!(replace_children(
    //     &call.syntax().parent().unwrap(),
    //     call.syntax().clone().into()..=call.syntax().clone().into(),
    //     Some(name.syntax().clone().into()),
    // ));
    // let name_ref = dbg!(node.descendants().find_map(ast::NameRef::cast).unwrap());
    // let ident = Ident::cast(tk.children_with_tokens().nth(1).unwrap().into_token().unwrap());
    // sema.scope(tk)
    //     .process_all_names(&mut |name, def| match def {
    //         hir::ScopeDef::Local(l) => {
    //             dbg!(l);
    //         }
    //         _ => {}
    //     });
    // let tk = call.token_tree().unwrap().syntax().ancestors().filter(|x| {
    // 	x.kind()
    // });
    // sema.scope_at_offset(syn, offset);
    // sema.scope_at_offset(token, offset)
    // dbg!(syn);
    ty.parse().unwrap()
}

fn ast_from_text<N: AstNode>(text: &str) -> N {
    let parse = SourceFile::parse(text);
    let node = match parse.tree().syntax().descendants().find_map(N::cast) {
        Some(it) => it,
        None => {
            panic!(
                "Failed to make ast node `{}` from text {}",
                std::any::type_name::<N>(),
                text
            )
        }
    };
    let node = node.syntax().clone();
    let node = unroot(node);
    let node = N::cast(node).unwrap();
    assert_eq!(node.syntax().text_range().start(), 0.into());
    node
}
fn unroot(n: SyntaxNode) -> SyntaxNode {
    SyntaxNode::new_root(n.green().to_owned())
}
fn line_col(line_col: proc_macro::LineColumn) -> LineCol {
    LineCol {
        line: line_col.line as u32 - 1,
        col: line_col.column as u32,
    }
}

fn load_workspace_at(
    root: AbsPathBuf,
    cargo_config: &CargoConfig,
    progress: &dyn Fn(String),
) -> Result<(AnalysisHost, vfs::Vfs, Option<ProcMacroClient>), Box<dyn Error>> {
    let root = ProjectManifest::discover_single(&root)?;
    let workspace = ProjectWorkspace::load(root, cargo_config, progress)?;

    load_workspace(workspace)
}

fn load_workspace(
    ws: ProjectWorkspace,
) -> Result<(AnalysisHost, vfs::Vfs, Option<ProcMacroClient>), Box<dyn Error>> {
    let (sender, receiver) = unbounded();
    let mut vfs = vfs::Vfs::default();
    let mut loader = {
        let loader =
            vfs_notify::NotifyHandle::spawn(Box::new(move |msg| sender.send(msg).unwrap()));
        Box::new(loader)
    };

    let proc_macro_client = None;

    let build_data = None;

    let crate_graph = ws.to_crate_graph(
        build_data.as_ref(),
        proc_macro_client.as_ref(),
        &mut |path: &AbsPath| {
            let contents = loader.load_sync(path);
            let path = vfs::VfsPath::from(path.to_path_buf());
            vfs.set_file_contents(path.clone(), contents);
            vfs.file_id(&path)
        },
    );

    let project_folders = ProjectFolders::new(&[ws], &[], build_data.as_ref());
    loader.set_config(vfs::loader::Config {
        load: project_folders.load,
        watch: vec![],
        version: 0,
    });

    let host = load_crate_graph(
        crate_graph,
        project_folders.source_root_config,
        &mut vfs,
        &receiver,
    );
    Ok((host, vfs, proc_macro_client))
}

fn load_crate_graph(
    crate_graph: CrateGraph,
    source_root_config: SourceRootConfig,
    vfs: &mut vfs::Vfs,
    receiver: &Receiver<vfs::loader::Message>,
) -> AnalysisHost {
    let lru_cap = std::env::var("RA_LRU_CAP")
        .ok()
        .and_then(|it| it.parse::<usize>().ok());
    let mut host = AnalysisHost::new(lru_cap);
    let mut analysis_change = Change::new();

    // wait until Vfs has loaded all roots
    for task in receiver {
        println!("here");
        match task {
            vfs::loader::Message::Progress {
                n_done,
                n_total,
                config_version: _,
            } => {
                if n_done == n_total {
                    break;
                }
            }
            vfs::loader::Message::Loaded { files } => {
                for (path, contents) in files {
                    vfs.set_file_contents(path.into(), contents);
                }
            }
        }
    }
    let changes = vfs.take_changes();
    for file in changes {
        if file.exists() {
            let contents = vfs.file_contents(file.file_id).to_vec();
            if let Ok(text) = String::from_utf8(contents) {
                analysis_change.change_file(file.file_id, Some(Arc::new(text)))
            }
        }
    }
    let source_roots = source_root_config.partition(&vfs);
    analysis_change.set_roots(source_roots);

    analysis_change.set_crate_graph(crate_graph);

    host.apply_change(analysis_change);
    host
}
#[derive(Default)]
pub(crate) struct ProjectFolders {
    pub(crate) load: Vec<vfs::loader::Entry>,
    pub(crate) watch: Vec<usize>,
    pub(crate) source_root_config: SourceRootConfig,
}

impl ProjectFolders {
    pub(crate) fn new(
        workspaces: &[ProjectWorkspace],
        global_excludes: &[AbsPathBuf],
        build_data: Option<&BuildDataResult>,
    ) -> ProjectFolders {
        let mut res = ProjectFolders::default();
        let mut fsc = FileSetConfig::builder();
        let mut local_filesets = vec![];

        for root in workspaces.iter().flat_map(|it| it.to_roots(build_data)) {
            let file_set_roots: Vec<VfsPath> =
                root.include.iter().cloned().map(VfsPath::from).collect();

            let entry = {
                let mut dirs = vfs::loader::Directories::default();
                dirs.extensions.push("rs".into());
                dirs.include.extend(root.include);
                dirs.exclude.extend(root.exclude);
                for excl in global_excludes {
                    if dirs.include.iter().any(|incl| incl.starts_with(excl)) {
                        dirs.exclude.push(excl.clone());
                    }
                }

                vfs::loader::Entry::Directories(dirs)
            };

            if root.is_member {
                res.watch.push(res.load.len());
            }
            res.load.push(entry);

            if root.is_member {
                local_filesets.push(fsc.len());
            }
            fsc.add_file_set(file_set_roots)
        }

        let fsc = fsc.build();
        res.source_root_config = SourceRootConfig {
            fsc,
            local_filesets,
        };

        res
    }
}
#[derive(Default, Debug)]
pub(crate) struct SourceRootConfig {
    pub(crate) fsc: FileSetConfig,
    pub(crate) local_filesets: Vec<usize>,
}

impl SourceRootConfig {
    pub(crate) fn partition(&self, vfs: &vfs::Vfs) -> Vec<SourceRoot> {
        let _p = profile::span("SourceRootConfig::partition");
        self.fsc
            .partition(vfs)
            .into_iter()
            .enumerate()
            .map(|(idx, file_set)| {
                let is_local = self.local_filesets.contains(&idx);
                if is_local {
                    SourceRoot::new_local(file_set)
                } else {
                    SourceRoot::new_library(file_set)
                }
            })
            .collect()
    }
}
