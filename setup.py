from setuptools import find_packages, setup
from typing import List

hyp_e_dot='-e .'
def get_requirements(file_path:str)->List[str]:
    """
    This Function will return the list of requirements

    """
    requirements=[]
    with open(file_path,"r") as f:
        requirements=f.readlines()
        requirements=[ r.replace("\n","") for r in requirements]
        if hyp_e_dot in requirements:
            requirements.remove(hyp_e_dot)
        
    return requirements



setup(
    name="MLOPS-1",
    version='0.0.1',
    author="Shambhuraj",
    author_email="shambhurajpatil11@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")

)