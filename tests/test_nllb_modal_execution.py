from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import hashlib
import json
import unittest
from unittest.mock import Mock, patch
import zipfile

from runners.nllb_modal_execution import run_training, unpack_inputs


class ModalCheckpointTests(unittest.TestCase):
    def test_another_condition_package_is_rejected_even_with_matching_run_argument(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);archive=root/'inputs.zip'
            with zipfile.ZipFile(archive,'w') as saved:saved.writestr('manifest.json','{}')
            worker=SimpleNamespace(verify_package=lambda path:{'run_id':'run','condition':'variant_a'})
            with patch('runners.nllb_modal_execution.load_worker',return_value=worker):
                with self.assertRaisesRegex(ValueError,'authorized condition'):
                    unpack_inputs(archive,root/'out','hash','run',lambda path:'hash','variant_b')

    def test_changed_archive_is_rejected_before_extraction(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);archive=root/'inputs.zip';archive.write_bytes(b'incomplete upload')
            with self.assertRaisesRegex(ValueError,'checksum differs'):
                unpack_inputs(archive,root/'out','wrong','run',lambda path:'actual')
            self.assertFalse((root/'out').exists())

    def test_traversal_or_symlink_entries_are_rejected(self):
        for kind in ('traversal','symlink'):
            with self.subTest(kind=kind), TemporaryDirectory() as temporary:
                root=Path(temporary);archive=root/'inputs.zip'
                with zipfile.ZipFile(archive,'w') as saved:
                    item=zipfile.ZipInfo('../escape' if kind=='traversal' else 'link')
                    if kind=='symlink':
                        item.create_system=3;item.external_attr=0o120777<<16
                    saved.writestr(item,b'/another/path')
                sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaisesRegex(ValueError,'Unsafe'):
                    unpack_inputs(archive,root/'out',sha(archive),'run',sha)
                self.assertFalse((root/'out').exists())

    def worker(self, root):
        run_id = 'fixture-run'
        def write(path, value):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value))
        w = SimpleNamespace(verify_package=lambda package:{'run_id':run_id},
                            verify_checkpoint=Mock(), sha=lambda path:'fixture-sha',
                            write_json=write, event=Mock())
        def commit(local, target, identity, step):
            directory = target/'checkpoints'/'step-1';directory.mkdir(parents=True)
            write(target/'latest_checkpoint.json',{'directory':'step-1','global_step':1})
            return directory
        w.commit_checkpoint = commit
        def run(package, runs, seconds, checkpoint_storage):
            w.commit_checkpoint(root/'local',runs/run_id,run_id,1)
            w.write_json(runs/run_id/'status.json',{'status':'training_complete'})
        w.run = run
        return w

    def test_commit_precedes_reported_persistent_checkpoint(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary);worker=self.worker(root)
            volume=SimpleNamespace(commit=Mock())
            def event(*args,**kwargs):
                if args[1]=='modal_volume_checkpoint_committed':
                    volume.commit.assert_called_once()
            worker.event.side_effect=event
            saved=run_training(worker,root/'package',root/'runs',3600,volume)
            self.assertEqual(saved['status']['checkpoint_storage'],'modal_volume')
            self.assertEqual(saved['checkpoint']['global_step'],1)

    def test_failed_volume_commit_propagates(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);worker=self.worker(root)
            original_event=worker.event
            volume=SimpleNamespace(commit=Mock(side_effect=OSError('Volume unavailable')))
            with self.assertRaisesRegex(OSError,'Volume unavailable'):
                run_training(worker,root/'package',root/'runs',3600,volume)
            original_event.assert_not_called()
