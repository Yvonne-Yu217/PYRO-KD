from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

from src.experiments import io


class ExperimentJSONLTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(directory) / "generations.jsonl"

    def test_torn_utf8_tail_is_backed_up_before_retry_can_be_appended(self):
        original = b'{"id": "done"}\n{"id": "retry", "output": "caf\xc3'
        self.path.write_bytes(original)

        self.assertEqual(io.read_jsonl(self.path), [{"id": "done"}])
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        self.assertEqual(self.path.read_bytes(), original)
        io.append_jsonl(self.path, [{"id": "retry", "output": "complete"}])

        self.assertEqual(io.read_jsonl(self.path), [{"id": "done"}, {"id": "retry", "output": "complete"}])
        backups = list(self.path.parent.glob("generations.jsonl.before-recovery-*.bin"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original)

    def test_valid_final_record_without_newline_gets_a_separator(self):
        original = '{"id": "café"}'.encode()
        self.path.write_bytes(original)

        self.assertEqual(io.read_jsonl(self.path), [{"id": "café"}])
        io.append_jsonl(self.path, [{"id": "new"}])

        self.assertEqual(self.path.read_bytes(), original + b'\n{"id": "new"}\n')
        self.assertEqual(io.read_jsonl(self.path), [{"id": "café"}, {"id": "new"}])

    def test_malformed_complete_records_are_reported_and_left_intact(self):
        for corrupt in (b'{"id": "done"}\n{\n{"id": "valid"}', b'{"id": "done"}\n{\n'):
            with self.subTest(contents=corrupt):
                self.path.write_bytes(corrupt)
                with self.assertRaisesRegex(ValueError, r"generations.jsonl: malformed JSONL at line 2"):
                    io.read_jsonl(self.path)
                self.assertEqual(self.path.read_bytes(), corrupt)

    def test_interrupted_append_does_not_discard_corrupt_interior_record(self):
        corrupt = b'{"id": "done"}\n{\n{"id":'
        self.path.write_bytes(corrupt)

        with self.assertRaisesRegex(ValueError, r"generations.jsonl: malformed JSONL at line 2"):
            io.append_jsonl(self.path, [{"id": "new"}])
        self.assertEqual(self.path.read_bytes(), corrupt)

    def test_append_to_complete_file_does_not_reparse_previous_records(self):
        self.path.write_bytes(b'{"id": "done"}\n')

        with patch.object(io, "_inspect_jsonl", side_effect=AssertionError("unexpected reparse")):
            io.append_jsonl(self.path, [{"id": "new"}])

        self.assertEqual(io.read_jsonl(self.path), [{"id": "done"}, {"id": "new"}])

    def test_final_non_object_json_is_reported_as_invalid(self):
        self.path.write_bytes(b'{"id": "done"}\n["not a generation"]')

        with self.assertRaisesRegex(ValueError, r"expected a JSON object at line 2"):
            io.read_jsonl(self.path)

    def test_failed_repair_leaves_original_and_backup_for_retry(self):
        original = b'{"id": "done"}\n{"id":'
        self.path.write_bytes(original)
        replace = io.os.replace

        def fail_repair(source, destination):
            if destination == self.path:
                raise OSError("publication failed")
            replace(source, destination)

        with patch.object(io.os, "replace", side_effect=fail_repair):
            with self.assertRaisesRegex(OSError, "publication failed"):
                io.append_jsonl(self.path, [{"id": "retry"}])
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])
        backups = list(self.path.parent.glob("generations.jsonl.before-recovery-*.bin"))
        self.assertEqual([backup.read_bytes() for backup in backups], [original])

        io.append_jsonl(self.path, [{"id": "retry"}])
        self.assertEqual(io.read_jsonl(self.path), [{"id": "done"}, {"id": "retry"}])

    @unittest.skipIf(io.fcntl is None, "requires POSIX file locking")
    def test_concurrent_append_waits_for_repair_and_keeps_both_rows(self):
        self.path.write_bytes(b'{"id": "done"}\n{"id":')
        publishing, release, second_started, second_read = (Event() for _ in range(4))
        write_bytes = io._write_bytes_atomic
        prepare_append = io._prepare_jsonl_append

        def block_repair(path, data, mode):
            if path == self.path:
                publishing.set()
                self.assertTrue(release.wait(5))
            write_bytes(path, data, mode)

        def observe_prepare_append(path):
            if second_started.is_set():
                second_read.set()
            return prepare_append(path)

        def append_second():
            second_started.set()
            io.append_jsonl(self.path, [{"id": "second"}])

        with (patch.object(io, "_write_bytes_atomic", side_effect=block_repair),
              patch.object(io, "_prepare_jsonl_append", side_effect=observe_prepare_append),
              ThreadPoolExecutor(2) as pool):
            first = pool.submit(io.append_jsonl, self.path, [{"id": "first"}])
            try:
                self.assertTrue(publishing.wait(5))
                second = pool.submit(append_second)
                self.assertTrue(second_started.wait(5))
                self.assertFalse(second_read.wait(0.1))
            finally:
                release.set()
            first.result(timeout=5)
            second.result(timeout=5)

        self.assertTrue(second_read.is_set())
        self.assertEqual(io.read_jsonl(self.path), [{"id": "done"}, {"id": "first"}, {"id": "second"}])
