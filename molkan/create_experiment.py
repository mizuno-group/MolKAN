import os
from datetime import datetime
import argparse

def create_experiment(template=None, subdir=None, subname=None):
    date_str = datetime.now().strftime("%y%m%d")
    abs_dir = os.path.dirname(os.path.abspath(__file__))
    if subdir is not None:
        os.makedirs(abs_dir + f"/experiments/{subdir}", exist_ok=True)
        if subname is not None:
            files = [f for f in os.listdir(abs_dir + f"/experiments/{subdir}") if f.startswith(f"{subname}_exp_")]
            next_exp_num = len(files) + 1
            new_file = abs_dir + f"/experiments/{subdir}/{subname}_exp_{date_str}_{next_exp_num:02}.py"
        else:
            files = [f for f in os.listdir(abs_dir + f"/experiments/{subdir}") if f.startswith("exp_")]
            next_exp_num = len(files) + 1
            new_file = abs_dir + f"/experiments/{subdir}/exp_{date_str}_{next_exp_num:02}.py"
    else:
        if subname is not None:
            files = [f for f in os.listdir(abs_dir + f"/experiments") if f.startswith(f"{subname}_exp_")]
            next_exp_num = len(files) + 1
            new_file = abs_dir + f"/experiments/{subname}_exp_{date_str}_{next_exp_num:02}.py"
        else:
            files = [f for f in os.listdir(abs_dir + f"/experiments") if f.startswith("exp_")]
            next_exp_num = len(files) + 1
            new_file = abs_dir + f"/experiments/exp_{date_str}_{next_exp_num:02}.py"

    if template is not None:
        with open(template, mode="r") as template_py:
            py_content = template_py.read()
        with open(template.replace(".py", ".sh"), mode="r") as template_sh:
            sh_content = template_sh.read()
    else:
        with open("experiments/template.py", "r") as template_py:
            py_content = template_py.read()
        with open("experiments/template.sh", mode="r") as template_sh:
            sh_content = template_sh.read()
            
    with open(new_file, "w") as new_exp:
        new_exp.write(py_content)
    with open(new_file.replace(".py", ".sh"), "w") as new_sh:
        new_sh.write(sh_content)
    print(f"New experiment created: {new_file.replace(".py", "")}")

if __name__ == "__main__":

    def parse_None_or_str(s):
        if s == "None":
            return None
        return str(s)

    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=parse_None_or_str, help="template file if you want to use other template temporary")
    parser.add_argument("--subdir", type=parse_None_or_str, help="directory under experiments if you need")
    parser.add_argument("--subname", type=parse_None_or_str, help="experiment file name if you need")
    args = parser.parse_args()

    create_experiment(args.template, args.subdir, args.subname)