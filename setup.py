from setuptools import setup, find_packages

setup(
    name="ai-exper",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "typer[all]",
        "pydantic>=1.9.0"
    ]
    entry_points={
        "console_scripts": [
            "ai-exper=ai_exper.cli:main"
        ]
    }
)
