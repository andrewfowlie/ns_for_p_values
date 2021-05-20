"""
Convert workflow to bash
========================
"""

from yaml import load, Loader


def print_steps(name):
    """
    Open workflow yaml and print build steps
    """
    with open(name) as stream:
        data = load(stream, Loader=Loader)

    for name, job in data["jobs"].items():

        print("# job: {}".format(name))
        print("# runs-on: {}".format(job["runs-on"]))

        for step in job["steps"]:
            if not step.get("run"):
                continue
            print("# {}".format(step.get("name")))
            print(step.get("run"))
  

if __name__ == "__main__":
    name = ".github/workflows/python-app.yml"
    print_steps(name)
