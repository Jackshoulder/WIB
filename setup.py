from setuptools import setup, find_packages

setup(
    name="WIB",
    version="1.0.0",
    author="Shuang Jiang, Tao Le",  
    description="Weak Interaction Barcode (WIB) for GROMACS trajectory analysis",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "pyyaml",
        "tqdm"
    ],
    entry_points={
        "console_scripts": [
            "wib-hbond = wib.analyze_hbonds:main",
            "wib-pipi = wib.analyze_pipi:main",
        ]
    },
    python_requires=">=3.8",
)
