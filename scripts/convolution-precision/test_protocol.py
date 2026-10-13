import copy,unittest
import analyze as a
class Contract(unittest.TestCase):
 def setUp(self):
  self.samples=a.P['bootstrap_samples'];a.P['bootstrap_samples']=100
  self.hosts=[]
  for cpu in a.P['required_cpus']:
   rows=[]
   for pair in range(32):
    for arm in ['candidate','published']:
     rows.append(dict(pair=pair,arm=arm,blocks=[dict(nanos=1000000000,calls=102,bytes=0) for _ in range(5)],measured=510,channels=10240,kernel=4,warmup=10000,sha256=a.P[arm+'_sha256'],hotNanosPerCall=80 if arm=='candidate' else 100,coldBytes=80 if arm=='candidate' else 100,hotBytesPerCall=0,coldOracleMismatches=0,hotOracleMismatches=0,coldHotMismatches=0,bits=128,fma=True))
   self.hosts.append(dict(cpu=cpu,rows=rows))
 def tearDown(self):a.P['bootstrap_samples']=self.samples
 def test_complete_strata(self):self.assertTrue(all(r['passed'] for r in a.assess(self.hosts)))
 def test_missing_cpu(self):
  with self.assertRaises(ValueError):a.assess(self.hosts[:-1])
 def test_partial_host(self):
  self.hosts[0]['rows'].pop()
  with self.assertRaises(AssertionError):a.assess(self.hosts)
 def test_duplicate_pair(self):
  self.hosts[0]['rows'][0]=copy.deepcopy(self.hosts[0]['rows'][2])
  with self.assertRaises(AssertionError):a.assess(self.hosts)
 def test_short_block(self):
  self.hosts[0]['rows'][0]['blocks'][0]['nanos']=999999999
  with self.assertRaises(AssertionError):a.assess(self.hosts)
 def test_partial_input_cycle(self):
  self.hosts[0]['rows'][0]['blocks'][0]['calls']=103
  with self.assertRaises(AssertionError):a.assess(self.hosts)
 def test_regression_fails_without_cross_cpu_rescue(self):
  for r in self.hosts[0]['rows']:
   if r['arm']=='candidate':r['hotNanosPerCall']=101
  report=a.assess(self.hosts);self.assertFalse(report[0]['passed']);self.assertTrue(report[1]['passed'])
 def test_resource_regression(self):
  self.hosts[0]['rows'][0]['hotBytesPerCall']=9
  self.assertFalse(a.assess(self.hosts)[0]['passed'])
 def test_changed_numeric_mode(self):
  self.hosts[0]['rows'][0]['bits']=256
  with self.assertRaises(AssertionError):a.assess(self.hosts)
if __name__=='__main__':unittest.main()
