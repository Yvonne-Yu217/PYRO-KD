import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('review_tool',ROOT/'review_tool.py')
tool=importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
sys.modules["review_tool"] = tool
espec=importlib.util.spec_from_file_location("export_hpc_inputs", ROOT/"export_hpc_inputs.py")
exporter=importlib.util.module_from_spec(espec)
espec.loader.exec_module(exporter)


def clips():
    return [{'clip_id':f'clip-{m}-{c}','movie_id':f'movie-{m}'} for m in range(10) for c in range(3)]


def rating(item_id, preference='X'):
    return {'item_id':item_id,'completed':True,'preference':preference,'evidence':'At 00:04 the action matches.',
            'independent':True, **{v:{**{c:4 for c in tool.CRITERIA},**{c:'No' for c in tool.COLLISIONS}} for v in ('X','Y')}}


class ReviewTests(unittest.TestCase):
    def test_balanced_reproducible_assignment(self):
        rows=tool.plan(clips(),12)
        self.assertEqual(rows,tool.plan(clips(),12))
        self.assertNotEqual(rows,tool.plan(clips(),13))
        self.assertEqual(len(rows),60)
        for name in tool.TEAM:
            assigned=[r for r in rows if name in r['reviewers']]
            self.assertEqual(len(assigned),24)
            for contrast in tool.CONTRASTS:
                self.assertEqual(sum(r['contrast']==contrast for r in assigned),8)
        for row in rows:
            self.assertEqual(len(set(row['reviewers'])),2)
            self.assertNotEqual(row['X'],row['Y'])
        for m in range(10):
            for contrast in tool.CONTRASTS:
                self.assertEqual(sum(r['movie_id']==f'movie-{m}' and r['contrast']==contrast for r in rows),2)
        for contrast,(a,_) in tool.CONTRASTS.items():
            self.assertEqual(sum(r['contrast']==contrast and r['X']==a for r in rows),10)

    def test_reject_duplicate_or_incomplete_membership(self):
        with self.assertRaises(ValueError): tool.plan(clips()[:-1],0)
        c=clips();c[1]['clip_id']=c[0]['clip_id']
        with self.assertRaises(ValueError): tool.plan(c,0)

    def test_complete_package_blinding_and_summary(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td); media=d/'source.mp4';media.write_bytes(b'fixture-video')
            records=clips()
            for c in records:
                c.update(original='source.mp4', variants={v:'source.mp4' for pair in tool.CONTRASTS.values() for v in pair})
            tool.dump(d/'clips.json',{'clips':records})
            tool.dump(d/'membership.json',{'test':{'video_ids':[c['clip_id'] for c in records]}})
            result=tool.prepare(d/'clips.json',d/'out','test-study',3,d/'membership.json')
            self.assertEqual(result['ratings_expected'],120)
            key_path=d/'out/coordinator-only/study-key.json'; key=tool.load(key_path)
            exports=[]
            for name in tool.TEAM:
                folder=d/'out/reviewers'/name
                p=tool.load(folder/'review-package.json')
                self.assertEqual(len(p['items']),24)
                self.assertFalse((folder/'study-key.json').exists())
                self.assertFalse((folder/'demo').exists())
                text=json.dumps(p)
                for label in ['AV_P0','A_cue_P0','movie_id','clip_id','contrast']:
                    self.assertNotIn(label,text)
                for item in p['items']:
                    for path in item['media'].values(): self.assertTrue((folder/path).is_file())
                export={k:p[k] for k in ['schema_version','study_id','package_id','reviewer','demo']}
                export['ratings']=[rating(i['id']) for i in p['items']]
                path=d/f'{name}.json';tool.dump(path,export);exports.append(path)
            summary=tool.summarize(key_path,exports,d/'summary.json')
            self.assertEqual(summary['completed_reviews'],120)
            self.assertEqual(summary['missing_completed'],[])
            self.assertTrue(all(c['raw_agreement']==1 for c in summary['contrasts'].values()))
            with self.assertRaises(ValueError): tool.summarize(key_path,exports+[exports[0]],d/'duplicate.json')
            bad=tool.load(exports[0]);bad['demo']=True;tool.dump(d/'practice.json',bad)
            with self.assertRaises(ValueError): tool.summarize(key_path,[d/'practice.json'],d/'bad.json')
            bad['demo']=False;bad['ratings'][0]['completed']=False;tool.dump(d/'draft.json',bad)
            partial=tool.summarize(key_path,[d/'draft.json'],d/'partial.json')
            self.assertEqual(partial['completed_reviews'],23)
            self.assertEqual(len(partial['missing_completed']),97)
            bad['ratings'][1]=rating(bad['ratings'][1]['item_id'],'Cannot assess')
            bad['ratings'][1].pop('X');bad['ratings'][1].pop('Y');tool.dump(d/'cant.json',bad)
            tool.summarize(key_path,[d/'cant.json'],d/'cant-summary.json')

    def test_hpc_export_checks_receipts_and_speed(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td); cache=d/'cache';tts=d/'tts';media=d/'source.mp4';media.write_bytes(b'media')
            ids=tool.load(ROOT.parent/'configs/reframed_splits.json')['test']['video_ids']
            sources={};results=[]
            for ident in ids:
                folder=cache/ident;folder.mkdir(parents=True)
                tool.dump(folder/'manifest.json',{'source':str(media)})
                sources[str(folder/'manifest.json')]=tool.digest(folder/'manifest.json')
            labels=sorted({v for pair in tool.CONTRASTS.values() for v in pair})
            for label in labels:
                selector,prior=label.rsplit('_',1);run='run-'+label
                folder=tts/run/'rendered';folder.mkdir(parents=True)
                wav=folder/'line.wav';wav.write_bytes(b'wave')
                row={'id':label+'-1','video_id':ids[0],'time':2,'tts_audio':str(wav),
                     'tts_audio_sha256':tool.digest(wav),'tts_playback_speed':1.0}
                rendered=folder/'rendered.jsonl';rendered.write_text(json.dumps(row)+'\n')
                sources[str(rendered)]=tool.digest(rendered)
                results.append({'selector':selector,'prior':prior,'model':'qwen','timeline':True,'run_id':run,'coverage':{'valid':1}})
            tool.dump(d/'report.json',{'sources':sources,'results':results})
            exporter.export(cache,tts,d/'export.json',d/'report.json')
            data=tool.load(d/'export.json')
            self.assertEqual(len(data['clips']),30)
            self.assertEqual(data['clips'][0]['narrations']['AV_P0'][0]['playback_speed'],1.2)
            self.assertEqual(data['clips'][1]['narrations']['AV_P0'],[])
            wav.write_bytes(b'changed')
            with self.assertRaises(ValueError):exporter.export(cache,tts,d/'bad.json',d/'report.json')

    def test_rating_validation(self):
        r=rating('a');tool.validate_rating(r)
        r['X']['accuracy']=True
        with self.assertRaises(ValueError):tool.validate_rating(r)
        r=rating('a');r['independent']=False
        with self.assertRaises(ValueError):tool.validate_rating(r)
        r=rating('a','Cannot assess');r['evidence']='  '
        with self.assertRaises(ValueError):tool.validate_rating(r)

if __name__=='__main__':unittest.main()
