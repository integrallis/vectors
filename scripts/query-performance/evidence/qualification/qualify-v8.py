import subprocess, os, json, hashlib
from pathlib import Path
os.environ["JAVA_HOME"]="/opt/query/jdk"
java="/opt/query/jdk/bin/java"
root=Path("/opt/query")
base=Path("/opt/archive/vectors-search/public-final-jars")
archive="/opt/archive/vectors-search"
candidate="/opt/query/candidate-v8-jars"
sha="e4b4706edb05f96b077a9092a737a26b32a58892"
common=["python3", "/opt/query/harness/run.py", str(base), candidate, archive]
def run(command, log):
 print(log,flush=True)
 with (root/log).open("w") as output:subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,check=True)
run(common+[str(root/"flat-v8-final"),"--rounds","3","--modes","flat","--baseline-sha","c43c316","--candidate-sha",sha],"flat-v8-final.log")
cp=":".join(str(p) for p in sorted(base.glob("*.jar")))
classes=root/"fixture-classes";classes.mkdir(exist_ok=True)
run(["/opt/query/jdk/bin/javac","--add-modules=jdk.incubator.vector","-cp",cp,"-d",str(classes),"/opt/query/harness/QueryPerformance.java","/opt/query/harness/BuildPublicFixture.java"],"public-glove-compile.log")
fixture=root/"public-glove-baseline"
command=[java,"--add-modules=jdk.incubator.vector","--enable-native-access=ALL-UNNAMED","-Xms3g","-Xmx3g","-cp",str(classes)+":"+cp,"BuildPublicFixture",archive+"/inputs/glove-100-angular-train.fbin",str(fixture)]
(root/"public-glove-build-command.json").write_text(json.dumps(command,indent=2))
run(command,"public-glove-build.log")
files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in fixture.rglob("*") if p.is_file()}
(root/"public-glove-fixture-hashes.json").write_text(json.dumps(files,indent=2))
run(common+[str(root/"hnsw-v8-final"),"--rounds","3","--modes","heap,mapped,public","--cpu","2","--public-glove",str(fixture),"--baseline-sha","c43c316","--candidate-sha",sha],"hnsw-v8-final.log")
for name in ["flat-v8-final","hnsw-v8-final"]:
 run(["python3","/opt/query/harness/audit.py",str(root/name)],name+"-audit.log")
print("Qualification runs and audits finished",flush=True)
