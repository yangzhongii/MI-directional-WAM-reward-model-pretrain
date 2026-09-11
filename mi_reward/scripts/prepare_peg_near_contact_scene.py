"""Derive a free-peg near-contact scene for reward-data collection."""
from __future__ import annotations
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--target',required=True);args=ap.parse_args()
    root=ET.parse(args.source).getroot(); eq=root.find('equality')
    if eq is not None:
        for weld in list(eq.findall("weld[@name='peg_grasp_weld']")): eq.remove(weld)
    root.set('model',root.get('model','peg')+'_near_contact')
    root.insert(0,ET.Comment('Near-contact reward-data scene: free peg, no kinematic qpos attachment.'))
    out=Path(args.target);out.parent.mkdir(parents=True,exist_ok=True);ET.indent(root,space='  ');out.write_text(ET.tostring(root,encoding='unicode')+'\n');print(f'wrote {out}')
if __name__=='__main__':main()
