"""Regression fixtures for structural errors missed by legacy manuscript gates."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('narrative',Path(__file__).resolve().parents[1]/'scripts/check_hierarchy_narrative.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_duplicate_subsection_caught():
    errors=module.heading_errors('## 4.1 Title\n### 4.1.1 A\n### 4.1.1 B\n')
    assert any('duplicate heading' in e for e in errors)

def test_missing_parent_and_gap_caught():
    errors=module.heading_errors('## 4.1 Title\n### 4.1.2 Gap\n### 4.2.1 Orphan\n')
    assert any('no parent' in e for e in errors)
    assert any('nonsequential' in e for e in errors)

def test_sequential_outline_accepted():
    assert module.heading_errors('## 4.1 Title\n### 4.1.1 A\n### 4.1.2 B\n## 4.2 Next\n')==[]

def test_all_members_of_object_reference_are_checked():
    assert module.object_references('Рисунки 3.20, 3.21 і 4.99; рисунки 3.20–3.22',r'[Рр]исун\w*')==['3.20','3.21','4.99','3.20','3.22']
