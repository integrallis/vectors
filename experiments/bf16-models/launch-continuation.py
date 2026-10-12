# Bonsai continuation; each immutable phase is bounded to ninety minutes/$4 and 22:32 UTC.
# Original launcher SHA256 cf06f4fdda0f42b657471ddec21a43d6fea8d88b63881fa8fd14dcccb96e5c78
"""Launch only this experiment's two bounded workers, with immutable payload and receipts."""
import datetime,hashlib,json,pathlib,subprocess,sys

root=pathlib.Path('/Users/briansam-bodden/Code/java-ai/projects/models-gdn-proof');out=root/'benchmark-results/2026-10-11-gdn-proof'
version=sys.argv[1] if len(sys.argv)>1 else 'v1'
archFilter=sys.argv[2] if len(sys.argv)>2 else 'both'
meta=json.loads((out/f'payload-{version}.json').read_text());runName=meta['runName']
receipt=out/f'hosts-{version}{"" if archFilter=="both" else "-"+archFilter}.json';assert not receipt.exists()
assert not (out/f'hosts-{version}.json').exists()
if archFilter=='both':assert not any(out.glob(f'hosts-{version}-*.json'))
def aws(*args):return json.loads(subprocess.check_output(['aws',*args,'--output','json']))
quota=aws('service-quotas','get-service-quota','--region','us-east-1','--service-code','ec2','--quota-code','L-1216C47A')['Quota']['Value']
active=aws('ec2','describe-instances','--region','us-east-1','--filters','Name=instance-state-name,Values=running,pending,shutting-down,stopping')['Reservations']
instances=[i for r in active for i in r['Instances'] if i.get('InstanceLifecycle') != 'spot']
types=sorted({i['InstanceType'] for i in instances});used=0
if types:
    cores={t['InstanceType']:t['VCpuInfo']['DefaultVCpus'] for t in aws('ec2','describe-instance-types','--region','us-east-1','--instance-types',*types)['InstanceTypes']}
    used=sum(cores[i['InstanceType']] for i in instances)
assert used+(32 if archFilter=='both' else 16)<=quota
configs=[('x86','m6a.4xlarge','x64','x86_64','ami-0045d7fc2ad003464'),('arm','m7g.4xlarge','aarch64','aarch64','ami-03dac6c363b33b131')]
configs=[c for c in configs if archFilter in ('both',c[0])];assert configs
rates={}
for arch,kind,*_ in configs:
    filters=[{'Type':'TERM_MATCH','Field':k,'Value':v} for k,v in {'instanceType':kind,'location':'US East (N. Virginia)','operatingSystem':'Linux','tenancy':'Shared','preInstalledSw':'NA','capacitystatus':'Used'}.items()]
    prices=aws('pricing','get-products','--region','us-east-1','--service-code','AmazonEC2','--filters',json.dumps(filters))['PriceList']
    values={float(d['pricePerUnit']['USD']) for p in prices for term in json.loads(p)['terms']['OnDemand'].values() for d in term['priceDimensions'].values() if d['unit']=='Hrs'}
    assert len(values)==1;rates[arch]=values.pop()
campaign_deadline=datetime.datetime(2026,10,12,22,32,tzinfo=datetime.timezone.utc)
duration=min(meta['maximumSecondsPerHost'],5400,int((campaign_deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds()));budget=min(meta['maximumCombinedUsd'],4.0)
assert duration>=3000, 'Insufficient time for declared continuation'
assert sum(rates.values())*duration/3600+.15<=budget
assert aws('s3api','list-objects-v2','--bucket','models-qual-077051030817','--prefix',f'results/{runName}/'+('' if archFilter=='both' else archFilter+'/'),'--max-keys','1').get('KeyCount',0)==0
fleet={'region':'us-east-1','lifecycle':'on-demand','maximumCombinedUsd':budget,'maximumSecondsPerHost':duration,'payload':meta['payloadUri'],
       'payloadSha256':meta['payloadSha256'],'quota':quota,'usedVcpus':used,'rates':rates,'hosts':[]}
receipt.write_text(json.dumps(fleet,indent=2)+'\n')
for arch,kind,jdkarch,awsarch,ami in configs:
    deadline=min(campaign_deadline,datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=duration))
    result=f's3://models-qual-077051030817/results/{runName}/{arch}/'
    jdk=meta['jdk'][jdkarch]
    body=f'''#!/bin/bash
set -euo pipefail
mkdir -p /work/out /opt/jdk
exec > >(tee -a /work/out/bootstrap.log /dev/console) 2>&1
remaining=$(( {int(deadline.timestamp())} - $(date +%s) ))
if [ "$remaining" -le 0 ]; then shutdown -h now; exit 1; fi
nohup bash -c "sleep $remaining; shutdown -h now" >/work/out/deadline.log 2>&1 &
export RESULT_URI={result}
finish() {{
 rc=$?
 if [ "$rc" -ne 0 ]; then echo "FAILED $rc" > /work/out/STATUS; fi
 if command -v aws >/dev/null; then aws s3 cp /work/out "$RESULT_URI" --recursive --only-show-errors || true; fi
 shutdown -h now
}}
trap finish EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl unzip python3 ca-certificates build-essential cmake time
curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-{awsarch}.zip -o /tmp/awscli.zip
unzip -qo /tmp/awscli.zip -d /tmp
/tmp/aws/install
cd /work
aws s3 cp {meta['payloadUri']} payload.tar.gz --only-show-errors
printf '%s  payload.tar.gz\n' {meta['payloadSha256']} | sha256sum --check --status
tar xzf payload.tar.gz
sha256sum --check --status inputs.sha256
cp inputs.sha256 provenance.json out/
aws s3 cp /work/out "$RESULT_URI" --recursive --only-show-errors
nohup bash -c 'while sleep 60; do aws s3 cp /work/out "$RESULT_URI" --recursive --only-show-errors; done' >out/upload.log 2>&1 &
curl -fL --retry 3 '{jdk['url']}' -o /tmp/jdk.tgz
printf '%s  /tmp/jdk.tgz\n' {jdk['sha256']} | sha256sum --check --status
tar xzf /tmp/jdk.tgz -C /opt/jdk --strip-components=1
curl -fsSL https://sh.rustup.rs -o /tmp/rustup.sh
sh /tmp/rustup.sh -y --profile minimal --default-toolchain 1.96.0
/opt/jdk/bin/java -version 2>out/java-version.txt
/root/.cargo/bin/rustc --version >out/rust-version.txt
lscpu >out/lscpu.txt
uname -a >out/uname.txt
ps aux >out/processes-before.txt
python3 run-host.py
'''
    userdata=out/f'{arch}-{version}-worker.sh';userdata.write_text(body);assert len(body.encode())<16384
    command=['ec2','run-instances','--region','us-east-1','--image-id',ami,'--instance-type',kind,
        '--key-name','qual-milan','--iam-instance-profile','Name=models-qual-worker',
        '--security-group-ids','sg-0e9ff523da4140d55','--subnet-id','subnet-08fcc910aba0d5b23',
        '--block-device-mappings',json.dumps([{'DeviceName':'/dev/sda1','Ebs':{'VolumeSize':40,'VolumeType':'gp3','DeleteOnTermination':True}}]),
        '--instance-initiated-shutdown-behavior','terminate','--metadata-options','HttpTokens=required,HttpEndpoint=enabled',
        '--user-data','file://'+str(userdata),'--tag-specifications',
        f'ResourceType=instance,Tags=[{{Key=Name,Value=models-{runName}-{arch}}},{{Key=project,Value=models-qualification}},{{Key=delete-after,Value={deadline.isoformat()}}}]']
    instance=aws(*command)['Instances'][0]
    fleet['hosts'].append({'arch':arch,'instanceId':instance['InstanceId'],'instanceType':kind,
        'deadline':deadline.isoformat(),'resultUri':result,'launchCommand':['aws',*command],
        'userDataSha256':hashlib.sha256(body.encode()).hexdigest()})
    receipt.write_text(json.dumps(fleet,indent=2)+'\n');print(arch,instance['InstanceId'],flush=True)
