from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT="-e ."
def get_require(file_path:str)->List[str]:
    '''
    this func will return list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
name="mylearningproject",
version="0.0.1",
author="Jay",
author_email="shashwatxiia1415@gmail.com",
packages=find_packages(),
installs_requires=get_require('requirements.txt')
)