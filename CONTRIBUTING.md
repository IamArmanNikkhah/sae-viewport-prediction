# Contributing to the SAE Viewport Prediction Project

First off, thank you for being a part of this team! This document outlines the process for contributing code to our project. Following these guidelines helps us keep the codebase clean, maintainable, and ensures we can all work together smoothly.

Our goal isn't just to build a working system, but to learn how to build software like a professional team.

## The Golden Rule: The Forking Workflow

To protect our main codebase, **we never commit directly to the `main` branch of the central repository.** All work will be done on your personal forks and submitted for review through a **Pull Request (PR)**. This is a standard workflow in both open-source and industry.

Here is the complete process, from getting the latest code to merging your feature.

### Step 1: Setup (One-Time Only)

You only need to do this once at the beginning of the project.

1.  **Fork the Repository:** Create your own copy of the central repository by clicking the "Fork" button on GitHub.
2.  **Clone Your Fork:** Clone your personal fork (not the central one) to your local machine. Replace `[YOUR_USERNAME]` with your GitHub username.
    ```bash
    git clone https://github.com/[YOUR_USERNAME]/sae-viewport-prediction.git
    cd sae-viewport-prediction
    ```
3.  **Configure the `upstream` Remote:** You need to tell Git where the central, "upstream" repository is so you can keep your fork updated.
    ```bash
    # This command adds a new remote named 'upstream' that points to the main project repo
    git remote add upstream https://github.com/IamArmanNikkhah/sae-viewport-prediction.git
    
    # Verify that it was added correctly
    git remote -v
    # You should see 'origin' (pointing to your fork) and 'upstream' (pointing to the central repo)
    ```

### Step 2: Sync Your Fork (Do This Before Every New Task)

Before you start writing code for a new feature, you must sync your fork's `main` branch with the central repository to get the latest changes from your teammates.

```bash
# 1. Make sure you are on your main branch
git switch main

# 2. Pull the latest changes from the 'upstream' (central) repository
git pull upstream main

# 3. Push these updates to your 'origin' (personal fork) to keep it in sync
git push origin main
Step 3: Create a Feature Branch

Never work directly on your main branch. For every new task (e.g., a feature, a bugfix), create a new, descriptively named branch.

code
Bash
download
content_copy
expand_less
# Branch off from your up-to-date main branch
git switch -c <branch-name>

# --- Good Branch Names ---
# git switch -c feature/data-loader
# git switch -c bugfix/slerp-edge-case
# git switch -c refactor/vmf-utils-stability
Step 4: Write Code & Commit Changes

Now you can work on your task. Make small, logical commits with clear messages.

Follow our Coding Conventions (see below).

Write clear commit messages (see below).

code
Bash
download
content_copy
expand_less
# Example:
git add src/data/loader.py
git commit -m "feat: Implement initial quaternion loading from dataset"
Step 5: Push Your Branch to Your Fork

When you're ready to share your work or open a pull request, push your feature branch to your personal fork (origin).

code
Bash
download
content_copy
expand_less
git push origin feature/data-loader
Step 6: Open a Pull Request (PR)

A Pull Request is a formal proposal to merge your work into the main project.

Go to your fork on GitHub (https://github.com/[YOUR_USERNAME]/sae-viewport-prediction).

You should see a yellow banner with your branch name. Click "Compare & pull request".

Base repository: IamArmanNikkhah/sae-viewport-prediction Base branch: main

Head repository: [YOUR_USERNAME]/sae-viewport-prediction Head branch: feature/data-loader

Write a clear title and description. Explain what the PR does, why it's needed, and how to test it. Use the template provided in the PR body.

Step 7: Participate in Code Review

Once a PR is open, another team member (or the mentor) will review it.

For the Author: Be open to feedback! The goal is to improve the code. Make the requested changes, commit them, and push to your branch. The PR will update automatically.

For the Reviewer: Be constructive and respectful. Explain your reasoning. The goal is collaboration, not criticism. Check for correctness, style, and clarity.

Step 8: Merge

Once the PR is approved and passes any automated checks, a project administrator will merge it into the central main branch. Your contribution is now officially part of the project! You can then safely delete your feature branch.

Coding Conventions

To ensure our codebase is consistent and readable, we will follow these standards.

Language: Python 3.8+

Style Guide: We follow PEP 8. Please configure your editor to follow this standard.

Code Formatter: We use the black code formatter to ensure uniform style with no arguments. Run it on your files before committing.

code
Bash
download
content_copy
expand_less
pip install black
black src/

Docstrings: All modules, classes, and functions should have a docstring explaining their purpose, arguments, and return values. We will use the Google Python Style Guide for docstrings.

code
Python
download
content_copy
expand_less
"""A brief one-line summary of the module or function.

A more detailed explanation of its purpose and implementation.

Args:
    arg1 (int): Description of the first argument.
    arg2 (str): Description of the second argument.

Returns:
    bool: Description of the return value.
"""
Commit Message Guidelines

We use the Conventional Commits specification. This makes our project history easy to read and helps automate changelogs.

Your commit message should be structured like this:

code
Code
download
content_copy
expand_less
<type>: <subject>

Common Types:

Type	Description
feat	A new feature (e.g., adding the SLERP function).
fix	A bug fix (e.g., fixing a NaN error in velocity calculation).
docs	Changes to documentation only (e.g., updating the README).
style	Changes that do not affect the meaning of the code (e.g., running black).
refactor	A code change that neither fixes a bug nor adds a feature.
test	Adding missing tests or correcting existing tests.
chore	Changes to the build process or auxiliary tools (e.g., updating .gitignore).

Example Commit Messages:

feat: Implement numerically stable A3 inversion

fix: Handle edge case in Rodrigues rotation for small angles

docs: Add section on code review to CONTRIBUTING.md

test: Add unit tests for VMFUtils special functions

Thank you for helping us build a high-quality project!
