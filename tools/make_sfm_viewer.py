#!/usr/bin/env python3
# reconstruction.json → self-contained SfM 웹뷰어(html). 외부 fetch 없음 → 로컬 더블클릭으로 열림.
# 사용: python make_sfm_viewer.py <reconstruction.json> <out.html> [제목]
import json, base64, sys
import numpy as np

src = sys.argv[1]; out = sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else "SfM Viewer"
rec = json.load(open(src))[0]

# 점군 (좌표 + 색)
xyz, rgb = [], []
for p in rec['points'].values():
    xyz.append(p['coordinates']); rgb.append(p['color'])
xyz = np.array(xyz, np.float32); rgb = np.clip(np.array(rgb), 0, 255).astype(np.uint8)
N = int(sys.argv[4]) if len(sys.argv) > 4 else 250000   # 0 = 전체(풀)
if N > 0 and len(xyz) > N:
    idx = np.linspace(0, len(xyz) - 1, N).astype(int)
    xyz, rgb = xyz[idx], rgb[idx]

# 카메라 위치 (angle-axis w2c → C = -R^T t)
cams = []
for s in rec['shots'].values():
    r = np.array(s['rotation'], float); t = np.array(s['translation'], float)
    th = np.linalg.norm(r)
    if th < 1e-8:
        R = np.eye(3)
    else:
        k = r / th
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    cams.append(-R.T @ t)
cams = np.array(cams, np.float32)

c = xyz.mean(0)
xyz = (xyz - c).astype(np.float32); cams = (cams - c).astype(np.float32)
b = lambda a: base64.b64encode(a.tobytes()).decode()

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>
<style>html,body{{margin:0;height:100%;overflow:hidden;background:#0b0e12;font-family:system-ui}}
#hud{{position:fixed;top:10px;left:12px;color:#cfd6df;font-size:13px;background:rgba(11,14,18,.75);padding:8px 12px;border-radius:8px;line-height:1.5}}</style></head>
<body><div id="hud"><b>{title}</b><br>{len(xyz):,} points · {len(cams)} cameras<br>drag=rotate · wheel=zoom · right-drag=pan</div>
<script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const dec=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0)).buffer;
const xyz=new Float32Array(dec("{b(xyz)}"));
const rgb=new Uint8Array(dec("{b(rgb)}"));
const cam=new Float32Array(dec("{b(cams)}"));
const scene=new THREE.Scene();
const cm=new THREE.PerspectiveCamera(60,innerWidth/innerHeight,.01,1000);cm.position.set(0,0,8);
const rn=new THREE.WebGLRenderer({{antialias:true}});rn.setSize(innerWidth,innerHeight);rn.setPixelRatio(devicePixelRatio);document.body.appendChild(rn.domElement);
const ctrl=new THREE.OrbitControls(cm,rn.domElement);ctrl.enableDamping=true;
const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(xyz,3));
const col=new Float32Array(rgb.length);for(let i=0;i<rgb.length;i++)col[i]=rgb[i]/255;
g.setAttribute('color',new THREE.BufferAttribute(col,3));
scene.add(new THREE.Points(g,new THREE.PointsMaterial({{size:.02,vertexColors:true}})));
const cg=new THREE.BufferGeometry();cg.setAttribute('position',new THREE.BufferAttribute(cam,3));
scene.add(new THREE.Points(cg,new THREE.PointsMaterial({{size:.12,color:0xff5a3c}})));
addEventListener('resize',()=>{{cm.aspect=innerWidth/innerHeight;cm.updateProjectionMatrix();rn.setSize(innerWidth,innerHeight)}});
(function loop(){{requestAnimationFrame(loop);ctrl.update();rn.render(scene,cm)}})();
</script></body></html>"""
open(out, 'w').write(html)
print(f"저장: {out}  ({len(xyz):,} pts, {len(cams)} cams, {len(html)//1024}KB)")
