import json,pathlib,subprocess,datetime,hashlib
out=pathlib.Path('/Users/briansam-bodden/Code/java-ai/projects/vectors/benchmark-results/2026-10-11-bf16-determinism/reduction-pilot')
models=pathlib.Path('/Users/briansam-bodden/Code/java-ai/projects/models/benchmark-results/2026-10-11-bonsai-production')
meta=json.loads((out/'isolated-payload.json').read_text());receipt=out/'isolated-hosts.json';assert not receipt.exists()
fleet={'maximumCombinedUsd':2,'maximumSecondsPerHost':3600,'payload':meta,'hosts':[]}
quota=json.loads(subprocess.check_output(['aws','service-quotas','get-service-quota','--region','us-east-1','--service-code','ec2','--quota-code','L-1216C47A']))['Quota']['Value']
active=json.loads(subprocess.check_output(['aws','ec2','describe-instances','--region','us-east-1','--filters','Name=instance-state-name,Values=running,pending,shutting-down,stopping']))['Reservations']
instances=[i for r in active for i in r['Instances']];types=sorted({i['InstanceType'] for i in instances});used=0
if types:
 cores={t['InstanceType']:t['VCpuInfo']['DefaultVCpus'] for t in json.loads(subprocess.check_output(['aws','ec2','describe-instance-types','--region','us-east-1','--instance-types',*types]))['InstanceTypes']};used=sum(cores[i['InstanceType']] for i in instances)
assert used+32<=quota
fleet['quotaCheck']={'quota':quota,'usedVcpus':used,'requestedVcpus':32};receipt.write_text(json.dumps(fleet,indent=2)+'\n')
for arch,instance_type,basefile,awsarch in [('x86','m6a.4xlarge','host.json','x86_64'),('arm','m7g.4xlarge','arm-neon-host.json','aarch64')]:
 filters=[{'Type':'TERM_MATCH','Field':k,'Value':v} for k,v in {'instanceType':instance_type,'location':'US East (N. Virginia)','operatingSystem':'Linux','tenancy':'Shared','preInstalledSw':'NA','capacitystatus':'Used'}.items()]
 prices=json.loads(subprocess.check_output(['aws','pricing','get-products','--region','us-east-1','--service-code','AmazonEC2','--filters',json.dumps(filters)]))['PriceList']
 rates=[float(d['pricePerUnit']['USD']) for p in prices for term in json.loads(p)['terms'].get('OnDemand',{}).values() for d in term['priceDimensions'].values() if d['unit']=='Hrs'];assert len(set(rates))==1 and rates[0]<0.8
 result=f's3://models-qual-077051030817/results/vectors-bf16-reduction-pilot-v1/{arch}/'
 existing=json.loads(subprocess.check_output(['aws','s3api','list-objects-v2','--bucket','models-qual-077051030817','--prefix',f'results/vectors-bf16-reduction-pilot-v1/{arch}/','--max-keys','1']));assert existing.get('KeyCount',0)==0
 now=datetime.datetime.now(datetime.timezone.utc);deadline=now+datetime.timedelta(seconds=3600);epoch=int(deadline.timestamp());jdkarch='x64' if arch=='x86' else 'aarch64'
 body=f"""#!/bin/bash
set -euo pipefail
mkdir -p /work/out /opt/jdk
exec > >(tee -a /work/out/bootstrap.log /dev/console) 2>&1
remaining=$(( {epoch} - $(date +%s) ))
if [ \"$remaining\" -le 0 ]; then shutdown -h now; exit 1; fi
nohup bash -c \"sleep $remaining; shutdown -h now\" >/work/out/deadline.log 2>&1 &
export RESULT_URI={result}
finish() {{
 rc=$?
 if [ \"$rc\" -ne 0 ]; then echo \"FAILED $rc\" > /work/out/STATUS; fi
 if command -v aws >/dev/null; then aws s3 cp /work/out \"$RESULT_URI\" --recursive --only-show-errors || true; fi
 shutdown -h now
}}
trap finish EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq unzip python3 ca-certificates
curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-{awsarch}.zip -o /tmp/awscli.zip
unzip -qo /tmp/awscli.zip -d /tmp
/tmp/aws/install
cd /work
aws s3 cp {meta['payloadUri']} payload.tar.gz --only-show-errors
printf '%s  payload.tar.gz\\n' {meta['payloadSha256']} | sha256sum --check --status
tar xzf payload.tar.gz
sha256sum --check --status inputs.sha256
cp inputs.sha256 provenance.json out/
aws s3 cp /work/out \"$RESULT_URI\" --recursive --only-show-errors
nohup bash -c 'while sleep 60; do aws s3 cp /work/out \"$RESULT_URI\" --recursive --only-show-errors; done' >/work/out/upload.log 2>&1 &
curl -fsSL 'https://api.adoptium.net/v3/assets/latest/25/hotspot?architecture={jdkarch}&image_type=jdk&os=linux&vendor=eclipse' -o out/jdk-release.json
python3 - <<'PYJDK'
import json,pathlib,urllib.request,hashlib
info=json.loads(pathlib.Path('out/jdk-release.json').read_text())[0]['binary']['package']
with urllib.request.urlopen(info['link']) as src,pathlib.Path('/tmp/jdk.tgz').open('wb') as dst:
 import shutil
 shutil.copyfileobj(src,dst)
assert hashlib.sha256(pathlib.Path('/tmp/jdk.tgz').read_bytes()).hexdigest()==info['checksum']
pathlib.Path('out/jdk.sha256').write_text(info['checksum']+'  /tmp/jdk.tgz\\n')
PYJDK
tar xzf /tmp/jdk.tgz -C /opt/jdk --strip-components=1
/opt/jdk/bin/java -version 2>out/java-version.txt
lscpu > out/lscpu.txt
uname -a > out/uname.txt
hostname > out/hostname.txt
ps aux > out/processes-before.txt
echo RUNNING_BF16_PAIRED > out/STATUS
python3 run-host.py
"""
 userdata=out/f'isolated-{arch}-worker.sh';userdata.write_text(body);assert len(body.encode())<16384
 command=list(json.loads((models/basefile).read_text())['launchCommand']);command[command.index('--user-data')+1]='file://'+str(userdata);command[command.index('--instance-type')+1]=instance_type;command[command.index('--block-device-mappings')+1]=json.dumps([{'DeviceName':'/dev/sda1','Ebs':{'VolumeSize':30,'VolumeType':'gp3','DeleteOnTermination':True}}]);command[command.index('--tag-specifications')+1]=f'ResourceType=instance,Tags=[{{Key=Name,Value=vectors-bf16-{arch}-reduction-pilot}},{{Key=project,Value=models-qualification}},{{Key=delete-after,Value={deadline.isoformat()}}}]'
 instance=json.loads(subprocess.check_output(command))['Instances'][0]
 fleet['hosts'].append({'arch':arch,'instanceId':instance['InstanceId'],'privateDnsName':instance['PrivateDnsName'],'instanceType':instance_type,'priceUsdPerHour':rates[0],'deadline':deadline.isoformat(),'resultUri':result,'userDataSha256':hashlib.sha256(body.encode()).hexdigest(),'launchCommand':command});receipt.write_text(json.dumps(fleet,indent=2)+'\n');print(arch,instance['InstanceId'],rates[0],flush=True)
