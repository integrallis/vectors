import copy,unittest
import analyze as a
class Contract(unittest.TestCase):
 def setUp(self):
  self.saved=copy.deepcopy(a.P);a.P['bootstrap_samples']=100;self.hosts={}
  for spec in a.P['strata']:
   for slot in range(spec['hosts']):
    rows={}
    for case,count in spec['counts'].items():
     method,shape=a.P['cases'][case]
     for pair in range(count):
      for arm in ['candidate','baseline']:
       rows[case,pair,arm]={'benchmark':'bench.'+method,'params':{'shape':shape},'primaryMetric':{'score':80 if arm=='candidate' else 100,'scoreUnit':'us/op'},'secondaryMetrics':{'gc.alloc.rate.norm':{'score':0}},'warmupIterations':5,'measurementIterations':5,'forks':1,'warmupTime':'1 s','measurementTime':'1 s','jdkVersion':'25.0.3','jvmArgs':['-XX:ActiveProcessorCount=4','-Xmx3g']}
    hardware={'stratum':spec['id'],'slot':slot,'cpu':spec['cpu'],'actual':{'useAVX':spec['useAVX'],'maxVectorSize':spec['maxVectorSize']}}
    self.hosts[spec['id'],slot]={'manifest':{'hardware':hardware},'rows':rows}
 def tearDown(self):a.P.clear();a.P.update(self.saved)
 def test_complete_exact_counts(self):
  r=a.assess(self.hosts);self.assertEqual([v['pairs'] for v in r],[128,256,256,64]);self.assertTrue(all(v['latency']['precisionTargetMet'] for v in r))
 def test_missing_host(self):
  self.hosts.pop(next(iter(self.hosts)))
  with self.assertRaises(AssertionError):a.assess(self.hosts)
 def test_missing_arm(self):
  next(iter(self.hosts.values()))['rows'].pop(('matvec_tall',0,'candidate'))
  with self.assertRaises(AssertionError):a.assess(self.hosts)
 def test_wrong_shape(self):
  next(iter(self.hosts.values()))['rows']['matvec_tall',0,'candidate']['params']['shape']='48x5120'
  with self.assertRaises(AssertionError):a.assess(self.hosts)
 def test_wrong_capability(self):
  next(iter(self.hosts.values()))['manifest']['hardware']['actual']['useAVX']=2
  with self.assertRaises(AssertionError):a.assess(self.hosts)
 def test_wrong_process_budget(self):
  next(iter(self.hosts.values()))['rows']['matvec_tall',0,'candidate']['jvmArgs']=['-XX:ActiveProcessorCount=8','-Xmx3g']
  with self.assertRaises(AssertionError):a.assess(self.hosts)
 def test_stable_host_order(self):self.assertEqual(a.assess(self.hosts),a.assess(dict(reversed(list(self.hosts.items())))))
 def test_regression_not_rescued(self):
  for (stratum,slot),h in self.hosts.items():
   if stratum=='9V45-avx512':
    for (case,pair,arm),r in h['rows'].items():
     if case=='matvec_small' and arm=='candidate':r['primaryMetric']['score']=102
  results=a.assess(self.hosts);self.assertEqual(results[1]['latency']['direction'],'slower');self.assertEqual(results[0]['latency']['direction'],'faster')
if __name__=='__main__':unittest.main()
