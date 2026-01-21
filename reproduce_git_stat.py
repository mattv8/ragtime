import os
import subprocess
import shutil

repo_dir = "test_repo_large_commit"

def run_git(args):
    subprocess.check_call(["git", "-C", repo_dir] + args)

def setup_repo():
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)
    os.makedirs(repo_dir)
    run_git(["init"])
    run_git(["config", "user.email", "test@example.com"])
    run_git(["config", "user.name", "Test User"])

    # Create 100 files
    for i in range(100):
        with open(os.path.join(repo_dir, f"file_{i}.txt"), "w") as f:
            f.write(f"content {i}\n")
    
    run_git(["add", "."])
    run_git(["commit", "-m", "Add many files"])

if __name__ == "__main__":
    setup_repo()
    
    # Run git show --stat with format
    cmd = [
        "git", "-C", repo_dir, "show", "HEAD", "--stat",
        "--format=Commit: %H%nAuthor: %an <%ae>%nDate: %ad%n%nSubject: %s%n%nBody:%n%b",
        "--date=iso"
    ]
    output = subprocess.check_output(cmd).decode("utf-8")
    print("--- Output Start ---")
    print(output)
    print("--- Output End ---")
