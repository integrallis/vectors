"""Freeze combined production BF16/activation source, controls, and full-model evidence harness."""
from pathlib import Path
import subprocess,shutil,hashlib,json,tarfile,zipfile,sys
repo=Path(__file__).resolve().parents[2];source=repo/'experiments/bf16-models'
central=repo.parent/'models-gdn-proof/benchmark-results/2026-10-11-gdn-proof'
version=sys.argv[1];assert version.startswith('bf16-models-v')
stage=Path('/tmp')/version;stage.mkdir(exist_ok=False)
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain','--','vectors-core'],cwd=repo,text=True).strip()
old=repo.parent/'models-gdn-proof/experiments/gdn/generated/payload-production-timing-v3'
candidate=repo/'vectors-core/build/libs/vectors-core-0.1.30.jar';assert candidate.is_file()
java=Path('/Users/briansam-bodden/.sdkman/candidates/java/25.0.3-tem/bin')
for arm in ['published','control','candidate']:
 shutil.copytree(old/'lib'/('published' if arm=='published' else 'native'),stage/'lib'/arm)
 if arm!='published':
  jars=list((stage/'lib'/arm).glob('vectors-core-*.jar'));assert len(jars)==1;jars[0].unlink()
  shutil.copy(candidate,stage/'lib'/arm/candidate.name)
# Keep the corrected control identical except for the original fixed BF16 implementation.
name='com/integrallis/vectors/core/BFloat16Matrix.class'
with zipfile.ZipFile('/tmp/vectors-core-bf16-final.jar') as z:control_class=z.read(name)
control_dir=stage/'control-override';(control_dir/name).parent.mkdir(parents=True);(control_dir/name).write_bytes(control_class)
subprocess.run([str(java/'jar'),'--update','--file',str(stage/'lib/control'/candidate.name),'-C',str(control_dir),name],check=True)
shutil.rmtree(control_dir)
for arm,original in [('published','published'),('corrected','native')]:shutil.copytree(old/'rust'/original,stage/'rust'/arm)
shutil.copytree(old/'data',stage/'data')
shutil.copytree(source,stage/'source',ignore=shutil.ignore_patterns('classes','*.log','*.jar','__pycache__'))
shutil.copy(source/'run-host.py',stage/'run-host.py')
(stage/'source/vectors-core.patch').write_bytes(subprocess.check_output(['git','diff','7e77a63',revision,'--','vectors-core'],cwd=repo))
(stage/'source/models-corrected.patch').write_bytes(subprocess.check_output(['git','diff','4337ee45','12796ebf','--','backend-native','backend-java'],cwd=repo.parent/'models-gdn-proof'))
script=(repo/'experiments/bf16-tiles/run-host.py').read_text()
script=script.replace('for pair in range(6):', 'for pair in range(12):')
script=script.replace("    except (AssertionError", "        run(arm+'-nested',common+['-cp',arm+'.jar:probes.jar:benchmark.jar','com.integrallis.vectors.core.BFloat16NestedDispatchProbe'])\n    except (AssertionError")
script=script.replace("(out/'STATUS').write_text('COMPLETE\\n')", "(out/'STATUS').write_text('KERNEL_COMPLETE\\n')")
(stage/'run-kernel-host.py').write_text(script)
shutil.copy(candidate,stage/'tile2parallel.jar')
shutil.copy('/tmp/vectors-bf16-final-expressions-v1/baseline.jar',stage/'baseline.jar')
shutil.copy('/tmp/vectors-bf16-final-expressions-v1/benchmark.jar',stage/'benchmark.jar')
classes=stage/'classes';classes.mkdir()
subprocess.run([str(java/'javac'),'-cp',str(stage/'lib/candidate/*'),'-d',str(classes),str(source/'Bf16ModelProof.java')],check=True)
subprocess.run([str(java/'jar'),'--create','--file',str(stage/'model-proof.jar'),'-C',str(classes),'.'],check=True)
shutil.rmtree(classes);classes.mkdir()
probes=['BFloat16MatrixColdHotProbe','BFloat16MatrixAllocationProbe','BFloat16NestedDispatchProbe']
subprocess.run([str(java/'javac'),'--add-modules','jdk.incubator.vector','-cp',str(candidate),'-d',str(classes),*[str(repo/'vectors-core/src/test/java/com/integrallis/vectors/core'/(name+'.java')) for name in probes]],check=True)
subprocess.run([str(java/'jar'),'--create','--file',str(stage/'probes.jar'),'-C',str(classes),'.'],check=True);shutil.rmtree(classes)
qwen=json.loads((repo.parent/'models/scripts/fleet/smoke-qwen-bf16.json').read_text())[0]
config={'arms':['published','control','candidate'],'vectorsRevision':revision,
 'modelsRevisions':{'published':'4337ee45','control':'12796ebf','candidate':'12796ebf'},
 'controlBf16Revision':'292c889','controlBf16ClassSha256':hashlib.sha256(control_class).hexdigest(),
 'models':[{'key':'qwen-bf16','entrypoint':'.','files':qwen['files']},
 {'key':'bonsai2','entrypoint':'model.gguf','files':[{'path':'model.gguf','uri':'https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/b072e1d3b35a0a630cece372c2127528e0994386/Ternary-Bonsai-2-27B-PTQ1_0.gguf','sha256':'53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3','sizeBytes':5946648928}]}]}
(stage/'model-proof.json').write_text(json.dumps(config,indent=2)+'\n')
meta=json.loads((central/'payload-bf16-tiles-v2.json').read_text());meta={k:meta[k] for k in ['jdk','rust']}
meta.update(runName='gdn-proof-'+version,sourceRepository='vectors',revision=revision,maximumSecondsPerHost=7200,maximumCombinedUsd=3.5,
 files={str(p.relative_to(stage)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(stage.rglob('*')) if p.is_file()})
(stage/'inputs.sha256').write_text(''.join(f'{sha}  {name}\n' for name,sha in meta['files'].items()))
(stage/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
archive=central/('gdn-proof-'+version+'.tar.gz');assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
 for p in sorted(stage.iterdir()):tar.add(p,arcname=p.name)
meta.update(payloadUri='s3://models-qual-077051030817/payload/gdn-proof-'+version+'.tar.gz',payloadSha256=hashlib.sha256(archive.read_bytes()).hexdigest())
(central/('payload-'+version+'.json')).write_text(json.dumps(meta,indent=2)+'\n')
subprocess.run(['aws','s3','cp',str(archive),meta['payloadUri'],'--only-show-errors'],check=True)
print(archive,meta['payloadSha256'],flush=True)
