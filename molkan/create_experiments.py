import os
from datetime import datetime
import argparse

def create_experiment(template=None, subdir=None, subname=None):
    date_str = datetime.now().strftime("%y%m%d")
    if subdir is not None:
        os.makedirs(f"experiments/{subdir}", exist_ok=True)
        if subname is not None:
            files = [f for f in os.listdir(f"experiments/{subdir}") if f.startswith(f"{subname}_exp_")]
            next_exp_num = len(files) + 1
            new_file = f"experiments/{subdir}/{subname}_exp_{next_exp_num:02}.py"
        else:
            files = [f for f in os.listdir(f"experiments/{subdir}") if f.startswith("exp_")]
            next_exp_num = len(files) + 1
            new_file = f"experiments/{subdir}/exp_{next_exp_num:02}.py"
    else:
        if subname is not None:
            files = [f for f in os.listdir(f"experiments") if f.startswith(f"{subname}_exp_")]
            next_exp_num = len(files) + 1
            new_file = f"experiments/{subname}_exp_{next_exp_num:02}.py"
        else:
            files = [f for f in os.listdir(f"experiments") if f.startswith("exp_")]
            next_exp_num = len(files) + 1
            new_file = f"experiments/exp_{next_exp_num:02}.py"

    if template is not None:
        with open(template, mode="r") as template:
            content = template.read()
    else:
        with open("experiments/template.py", "r") as template:
            content = template.read()
            
    with open(new_file, "w") as new_exp:
        new_exp.write(content)
    print(f"New experiment created: {new_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", help="template file if you want to use other template temporary")
    parser.add_argument("--subdir", help="directory under experiments if you need")
    parser.add_argument("--subname", help="experiment file name if you need")
    args = parser.parse_args()

    create_experiment(args.template, args.subdir, args.subname)