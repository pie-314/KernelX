use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub struct SubmissionCheck {
    pub label: &'static str,
    pub status: CheckStatus,
    pub detail: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckStatus {
    Ready,
    Warning,
    Missing,
}

#[derive(Clone)]
pub struct RepoScan {
    pub checks: Vec<SubmissionCheck>,
    pub command_cards: Vec<&'static str>,
    pub readme_has_space_url: bool,
    pub readme_has_colab_url: bool,
    pub readme_has_repo_url: bool,
    pub readme_has_story_url: bool,
}

pub fn scan(root: &Path) -> RepoScan {
    let readme_path = root.join("README.md");
    let readme = fs::read_to_string(&readme_path).unwrap_or_default();
    let has_openenv_manifest = root.join("brain/openenv.yaml").exists();
    let has_training_entry = root.join("brain/server/run_autonomous.py").exists();
    let has_fastapi_entry = root.join("brain/server/app.py").exists();
    let has_ui_binary = root.join("ui/Cargo.toml").exists();
    let has_notebook = has_file_with_extension(root, "ipynb");
    let has_plot_asset = has_file_with_extension(root, "png") || has_file_with_extension(root, "jpg");

    let has_space_url = readme.contains("huggingface.co/spaces");
    let has_colab_url =
        readme.contains("colab.research.google.com") || readme.contains(".ipynb");
    let has_repo_url = readme.contains("github.com");
    let has_story_url = readme.contains("youtube.com")
        || readme.contains("youtu.be")
        || readme.contains("huggingface.co/blog");

    let checks = vec![
        SubmissionCheck {
            label: "README",
            status: if readme_path.exists() {
                CheckStatus::Ready
            } else {
                CheckStatus::Missing
            },
            detail: if readme_path.exists() {
                "Present at repo root".to_string()
            } else {
                "Missing the required submission index".to_string()
            },
        },
        SubmissionCheck {
            label: "OpenEnv manifest",
            status: if has_openenv_manifest {
                CheckStatus::Ready
            } else {
                CheckStatus::Missing
            },
            detail: "Judges require an OpenEnv-compliant environment".to_string(),
        },
        SubmissionCheck {
            label: "Training entrypoint",
            status: if has_training_entry {
                CheckStatus::Ready
            } else {
                CheckStatus::Missing
            },
            detail: "KernelX can run autonomously through the manual policy loop".to_string(),
        },
        SubmissionCheck {
            label: "Notebook / Colab",
            status: if has_notebook || has_colab_url {
                CheckStatus::Ready
            } else {
                CheckStatus::Warning
            },
            detail: if has_notebook {
                "Notebook asset exists in the repo".to_string()
            } else if has_colab_url {
                "README already references a Colab notebook".to_string()
            } else {
                "Required by the submission form; add a Colab link or notebook".to_string()
            },
        },
        SubmissionCheck {
            label: "HF Space URL",
            status: if has_space_url {
                CheckStatus::Ready
            } else {
                CheckStatus::Warning
            },
            detail: "README must link to the deployed environment".to_string(),
        },
        SubmissionCheck {
            label: "Story asset",
            status: if has_story_url {
                CheckStatus::Ready
            } else {
                CheckStatus::Warning
            },
            detail: "Need a YouTube video or Hugging Face blog post".to_string(),
        },
        SubmissionCheck {
            label: "Plots / media",
            status: if has_plot_asset {
                CheckStatus::Ready
            } else {
                CheckStatus::Warning
            },
            detail: "Readable reward plots should be committed for judges".to_string(),
        },
        SubmissionCheck {
            label: "Demo stack",
            status: if has_fastapi_entry && has_ui_binary {
                CheckStatus::Ready
            } else {
                CheckStatus::Missing
            },
            detail: "Environment server and TUI are both wired into the repo".to_string(),
        },
    ];

    RepoScan {
        checks,
        command_cards: vec![
            "1. make -C kernel load",
            "2. cargo run --manifest-path bridge/Cargo.toml --release",
            "3. cd brain && python3 -m server.app",
            "4. cd brain/server && python3 run_autonomous.py --steps 50 --verbose",
            "5. cargo run --manifest-path ui/Cargo.toml",
        ],
        readme_has_space_url: has_space_url,
        readme_has_colab_url: has_colab_url,
        readme_has_repo_url: has_repo_url,
        readme_has_story_url: has_story_url,
    }
}

fn has_file_with_extension(root: &Path, extension: &str) -> bool {
    let mut stack = vec![PathBuf::from(root)];
    while let Some(path) = stack.pop() {
        let entries = match fs::read_dir(&path) {
            Ok(entries) => entries,
            Err(_) => continue,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().and_then(|value| value.to_str()).unwrap_or("");
                if matches!(name, ".git" | "target" | "venv" | "__pycache__") {
                    continue;
                }
                stack.push(path);
                continue;
            }

            if path
                .extension()
                .and_then(|value| value.to_str())
                .map(|value| value.eq_ignore_ascii_case(extension))
                .unwrap_or(false)
            {
                return true;
            }
        }
    }
    false
}
