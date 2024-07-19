from setuptools import setup, find_packages
import os

# make the full path to README.md
readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md')

with open(readme_path, "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name = 'textagon',
    version = '0.1.19',
    packages = find_packages(),
    description = 'Textagon is a powerful tool for text data analysis, providing a means to visualize parallel representations of your data and gain insight into the impact of various lexicons on two classes of text data.',
    long_description=long_description,
    long_description_content_type='text/markdown', 
    author = 'Ahmed Abbasi, David Dobolyi, John Lalor, Ruiyang Qin',
    author_email = 'aabbasi@nd.edu, david.dobolyi@colorado.edu,  john.lalor@nd.edu, rqin@nd.edu',
    classifiers=[
        "License :: OSI Approved :: Python Software Foundation License"
    ],
    python_requires='>=3.6',
    install_requires=[
        'nltk >= 3.8.1',
        'scikit-learn >= 1.2.1',
        'pandas >= 1.5.3',
        'numpy >= 1.24.2',
        'beautifulsoup4 >= 4.11.2',
        'mapply >= 0.1.21',
        'pyenchant >= 3.2.2',
        'tzlocal >= 4.3',
        'spacy >= 3.5.2',
    ],
    entry_points={
        'console_scripts': [
            'textagon_post_install = textagon.post_install:main',
        ],
    },
    package_data={
        'textagon': [
            'external/lexicons/exclusions.txt',
            'external/extracted/WNAffect-master/**/*',
            'external/extracted/wordnet-1.6/**/*',
            'external/extracted/wn-domains/**/*',
            'external/extracted/wn-domains/wn-affect-1.1/a-synsets.xml',
            'external/extracted/wn-domains/wn-affect-1.1/a-hierarchy.xml',
            'external/lexicons/Lexicons_v5.zip',
            ],
    }
)
