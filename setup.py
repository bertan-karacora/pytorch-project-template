from pathlib import Path
import setuptools

# Package metadata
NAME_PACKAGE = "pytorch-project-template"
VERSION = "0.1.0"
DESCRIPTION = "A Python package that serves as a template for deep learning projects using Pytorch."
MAINTAINER = "Bertan Karacora"
EMAIL_MAINTAINER = "bertan.karacora@gmail.com"
NAME_DIR_PACKAGE = "project"

# Automatically find scripts in the scripts/ directory
PATH_DIR_SCRIPTS = Path(__file__).parent / NAME_DIR_PACKAGE / "scripts"
ENTRYPOINTS = []
if PATH_DIR_SCRIPTS.exists():
    for path_script in PATH_DIR_SCRIPTS.glob("*.py"):
        name_script = path_script.stem

        # Skip private scripts and __init__.py if it exists
        if name_script.startswith("_"):
            continue

        ENTRYPOINTS.append(f"{name_script} = {NAME_DIR_PACKAGE}.scripts.{name_script}:main")

# Setup the package
setuptools.setup(
    name=NAME_PACKAGE,
    version=VERSION,
    description=DESCRIPTION,
    long_description=(Path(__file__).parent / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url=f"https://github.com/bertan-karacora/{NAME_PACKAGE}",
    license="MIT",
    maintainer=MAINTAINER,
    maintainer_email=EMAIL_MAINTAINER,
    packages=setuptools.find_namespace_packages(include=[f"{NAME_DIR_PACKAGE}*", f"{NAME_DIR_PACKAGE}.*"]),
    include_package_data=True,
    entry_points={"console_scripts": ENTRYPOINTS},
)
