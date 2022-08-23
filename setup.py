from setuptools import setup, find_packages

setup(
    name="ai-exper",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "ai-exper=ai_exper.cli:main"
        ]
    }
)
