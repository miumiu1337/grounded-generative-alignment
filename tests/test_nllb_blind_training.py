import json
import hashlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile
from unittest.mock import patch

from preprocessing import finewiki_pilot as data
from runners import nllb_blind_training as worker
from runners import nllb_local_recovery as recovery
from runners.prepare_nllb_blind import normalize, benchmark_key, near_duplicate


class CheckpointTests(unittest.TestCase):
    def checkpoint(self, root, step):
        path = root / f'checkpoint-{step}'
        path.mkdir()
        for name in ('optimizer.pt', 'scheduler.pt', 'config.json', 'model.safetensors', 'rng_state.pth'):
            (path / name).write_text('fixture ' + name)
        (path / 'trainer_state.json').write_text(json.dumps({'global_step': step}))
        return path

    def test_successful_checkpoint_is_verified_before_old_copy_is_removed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); drive = root / 'drive'; drive.mkdir()
            first = worker.commit_checkpoint(self.checkpoint(root, 1), drive, 'run', 1)
            self.assertEqual(worker.verify_checkpoint(first, 'run')['global_step'], 1)
            second = worker.commit_checkpoint(self.checkpoint(root, 2), drive, 'run', 2)
            self.assertFalse(first.exists())
            self.assertEqual(worker.verify_checkpoint(second, 'run')['global_step'], 2)

    def test_failed_copy_preserves_previous_committed_checkpoint(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); drive = root / 'drive'; drive.mkdir()
            first = worker.commit_checkpoint(self.checkpoint(root, 1), drive, 'run', 1)
            with patch.object(worker.shutil, 'copyfile', side_effect=OSError('fixture Drive outage')):
                with self.assertRaisesRegex(OSError, 'Drive outage'):
                    worker.commit_checkpoint(self.checkpoint(root, 2), drive, 'run', 2)
            self.assertEqual(worker.verify_checkpoint(first, 'run')['global_step'], 1)
            self.assertEqual(json.loads((drive / 'latest_checkpoint.json').read_text())['global_step'], 1)

    def test_tampered_checkpoint_stops_resume(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); drive = root / 'drive'; drive.mkdir()
            saved = worker.commit_checkpoint(self.checkpoint(root, 1), drive, 'run', 1)
            (saved / 'optimizer.pt').write_text('corrupted')
            with patch.object(worker, 'event'):
                with self.assertRaisesRegex(ValueError, 'integrity failure'):
                    worker.restore_checkpoint(drive, root / 'local', 'run')

    def test_partial_checkpoint_is_ignored_and_valid_copy_restores(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); drive = root / 'drive'; drive.mkdir()
            worker.commit_checkpoint(self.checkpoint(root, 1), drive, 'run', 1)
            (drive / 'checkpoints/step-9999999-partial').mkdir()
            with patch.object(worker, 'event') as event:
                restored = worker.restore_checkpoint(drive, root / 'local', 'run')
            self.assertEqual(json.loads((restored / 'trainer_state.json').read_text())['global_step'], 1)
            self.assertIn('incomplete_checkpoint_ignored', [call.args[1] for call in event.call_args_list])

    def test_checkpoint_path_traversal_is_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); drive = root / 'drive'; drive.mkdir()
            saved = worker.commit_checkpoint(self.checkpoint(root, 1), drive, 'run', 1)
            receipt = json.loads((saved / 'complete.json').read_text())
            receipt['files']['../secret'] = 'invalid'
            (saved / 'complete.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'integrity failure'):
                worker.verify_checkpoint(saved, 'run')

    def test_manifest_and_data_tampering_are_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); (root / 'train.jsonl').write_text('fixture')
            manifest = {'files': {'train.jsonl': worker.sha(root / 'train.jsonl')}}
            manifest['run_id'] = data.digest(manifest)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            worker.verify_package(root)
            (root / 'train.jsonl').write_text('modified')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                worker.verify_package(root)

    def test_duplicate_key_is_unicode_and_whitespace_aware(self):
        self.assertEqual(normalize('  GRÜAß  DI\n'), normalize('grüaß di'))
        self.assertEqual(normalize('a\u0308'), normalize('ä'))

    def test_benchmark_overlap_ignores_punctuation_and_detects_small_edits(self):
        self.assertEqual(benchmark_key('Griaß di!'), benchmark_key('GRIAß DI.'))
        text = 'this is a sufficiently long sentence describing a historical event in bavaria'
        self.assertTrue(near_duplicate(text, text.replace('event', 'events')))
        self.assertFalse(near_duplicate(text, 'an entirely unrelated sentence with different words and subject matter'))
        self.assertFalse(near_duplicate('short text', 'short texts'))

    def local_checkpoint(self, root, step):
        directory = self.checkpoint(root, step)
        header = json.dumps({'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
        (directory / 'model.safetensors').write_bytes(len(header).to_bytes(8, 'little') + header + bytes(4))
        return directory

    def test_local_recovery_prefers_newer_save_and_preserves_drive_backup(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); drive = root / 'drive'; drive.mkdir()
            original = worker.commit_checkpoint(self.checkpoint(root, 400), drive, 'run', 400)
            pointer = (drive / 'latest_checkpoint.json').read_bytes()
            local = root / 'local'; local.mkdir()
            saved = self.local_checkpoint(local, 500)
            worker.seal_local_checkpoint(saved, 'run')
            with patch.object(worker, 'event'):
                restored = worker.restore_checkpoint(drive, local, 'run', prefer_local=True)
            self.assertEqual(restored, saved)
            self.assertEqual((drive / 'latest_checkpoint.json').read_bytes(), pointer)
            self.assertEqual(worker.verify_checkpoint(original, 'run')['global_step'], 400)

    def test_local_seal_rejects_truncated_weights(self):
        with TemporaryDirectory() as temporary:
            directory = self.local_checkpoint(Path(temporary), 500)
            path = directory / 'model.safetensors'
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, 'Truncated'):
                worker.seal_local_checkpoint(directory, 'run')
            self.assertFalse((directory / 'complete.json').exists())

    def test_local_seal_does_not_include_its_own_receipt(self):
        with TemporaryDirectory() as temporary:
            directory = self.local_checkpoint(Path(temporary), 500)
            first = worker.seal_local_checkpoint(directory, 'run')
            second = worker.seal_local_checkpoint(directory, 'run')
            self.assertEqual(first['files'], second['files'])
            self.assertNotIn('complete.json', second['files'])

    def test_local_backup_parts_reconstruct_a_verified_resumable_archive(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); local_base = root / 'local'
            inputs = root / 'inputs'; inputs.mkdir()
            manifest = {'files': {}}
            manifest['run_id'] = data.digest(manifest)
            (inputs / 'manifest.json').write_text(json.dumps(manifest))
            local = local_base / manifest['run_id'][:16]; local.mkdir(parents=True)
            directory = self.local_checkpoint(local, 500)
            worker.seal_local_checkpoint(directory, manifest['run_id'])
            with patch.object(recovery, 'Path', return_value=local_base):
                index = recovery.export_parts(Path(worker.__file__), inputs, root / 'exports')
            record = json.loads(index.read_text())
            raw = b''.join((index.parent / part['name']).read_bytes() for part in record['parts'])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), record['archive_sha256'])
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                self.assertIsNone(archive.testzip())
                self.assertIn('optimizer.pt', archive.namelist())
                self.assertEqual(json.loads(archive.read('complete.json'))['global_step'], 500)

    def test_single_zip_preserves_weights_and_restores_the_same_method(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);inputs=root/'inputs';inputs.mkdir()
            manifest={'condition':'laser','files':{}}
            manifest['run_id']=data.digest(manifest)
            (inputs/'manifest.json').write_text(json.dumps(manifest))
            local_base=root/'first';local=local_base/manifest['run_id'][:16];local.mkdir(parents=True)
            checkpoint=self.local_checkpoint(local,500)
            receipt=worker.seal_local_checkpoint(checkpoint,manifest['run_id'])
            with patch.object(recovery,'Path',side_effect=lambda value: local_base if str(value)=='/content/nllb_training_work' else Path(value)):
                archive=recovery.export_single(Path(worker.__file__),inputs,root/'exports')
            with zipfile.ZipFile(archive) as saved:
                self.assertIsNone(saved.testzip())
                self.assertEqual(json.loads(saved.read('training_manifest.json')),manifest)
                self.assertEqual(saved.read('model.safetensors'),(checkpoint/'model.safetensors').read_bytes())
            second=root/'second'
            with patch.object(recovery,'Path',side_effect=lambda value: second if str(value)=='/content/nllb_training_work' else Path(value)):
                restored=recovery.restore_single(Path(worker.__file__),inputs,archive)
            self.assertEqual(worker.verify_checkpoint(restored,manifest['run_id'])['files'],receipt['files'])
            persistent=root/'persistent-volume'
            copied=recovery.restore_single(Path(worker.__file__),inputs,archive,local_base=persistent)
            self.assertEqual(copied,persistent/manifest['run_id'][:16]/'checkpoint-500')
            self.assertEqual(worker.verify_checkpoint(copied,manifest['run_id'])['files'],receipt['files'])

    def test_single_zip_restore_rejects_another_method(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);inputs=root/'inputs';inputs.mkdir()
            manifest={'condition':'bge_m3','files':{}};manifest['run_id']=data.digest(manifest)
            (inputs/'manifest.json').write_text(json.dumps(manifest))
            archive=root/'other.zip'
            with zipfile.ZipFile(archive,'w') as saved:
                saved.writestr('training_manifest.json',json.dumps({'condition':'laser'}))
                saved.writestr('complete.json','{}')
            with patch.object(recovery,'Path',side_effect=lambda value: root/'local' if str(value)=='/content/nllb_training_work' else Path(value)):
                with self.assertRaisesRegex(ValueError,'another method'):
                    recovery.restore_single(Path(worker.__file__),inputs,archive)

    def test_single_zip_restore_rejects_oversized_metadata_before_decoding(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);inputs=root/'inputs';inputs.mkdir()
            manifest={'condition':'laser','files':{}};manifest['run_id']=data.digest(manifest)
            (inputs/'manifest.json').write_text(json.dumps(manifest))
            archive=root/'oversized.zip'
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as saved:
                saved.writestr('training_manifest.json','x'*(256*1024+1))
                saved.writestr('complete.json','{}')
            with patch.object(recovery,'Path',side_effect=lambda value: root/'local' if str(value)=='/content/nllb_training_work' else Path(value)):
                with self.assertRaisesRegex(ValueError,'Oversized'):
                    recovery.restore_single(Path(worker.__file__),inputs,archive)


if __name__ == '__main__':
    unittest.main()
