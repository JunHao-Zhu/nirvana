import os

query_ids = range(1, 13)
program_name_template = "workloads/imdb/q{qid}"

for qid in query_ids:
    program_name = program_name_template.format(qid=qid)
    os.system(f"python {program_name}.py")
