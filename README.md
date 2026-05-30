## Getting started

Clone the repo and follow the setup instructions before running anything.

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

## Setup Instructions

After cloning the repo, run these two commands:

**1. Install nbstripout to keep notebook diffs clean:**

```bash
pip install nbstripout
nbstripout --install
```

**2. Add a `.gitattributes` file at the repo root** with the following content:

```
* text=auto eol=lf
*.ipynb filter=nbstripout
```