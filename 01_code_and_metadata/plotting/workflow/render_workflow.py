#!/usr/bin/env python3
"""Render the workflow as editable SVG/Draw.io and PDF/PNG."""
from pathlib import Path
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import findfont, FontProperties

HERE = Path(__file__).resolve().parent
spec = yaml.safe_load((HERE/'figure.yaml').read_text())
cfg = spec['rendering']
W,H=cfg['width_mm'],cfg['height_mm']
font=cfg['font_family']
font_pt=cfg['font_size_pt']
svg_ns='http://www.w3.org/2000/svg'
ET.register_namespace('',svg_ns)
def se(tag,attrs=None,parent=None):
    return ET.SubElement(parent if parent is not None else svg, '{%s}%s'%(svg_ns,tag), {str(k):str(v) for k,v in (attrs or {}).items()})
svg=ET.Element('{%s}svg'%svg_ns,{'width':f'{W}mm','height':f'{H}mm','viewBox':f'0 0 {W} {H}','role':'img'})
se('title').text='Image-guided exposure planning and separate segmentation benchmarks'
se('desc').text=spec['figure']['caption']
se('rect',{'width':W,'height':H,'fill':'#FFFFFF'})
defs=se('defs'); marker=se('marker',{'id':'arrow','markerWidth':5,'markerHeight':5,'refX':4.1,'refY':2.5,'orient':'auto','markerUnits':'userSpaceOnUse'},defs)
se('path',{'d':'M0,0 L4.5,2.5 L0,5 Z','fill':'#687789'},marker)

plt.rcParams.update({'font.family':font,'font.size':font_pt,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(W/25.4,H/25.4),facecolor='white')
ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,W);ax.set_ylim(H,0);ax.axis('off')

mx=ET.Element('mxfile',{'host':'app.diagrams.net','type':'device','version':'24.7.17'})
dia=ET.SubElement(mx,'diagram',{'id':'Figure2A','name':'Image-guided exposure planning'})
model=ET.SubElement(dia,'mxGraphModel',{'dx':'692','dy':'299','grid':'1','gridSize':'4','page':'1','pageScale':'1','pageWidth':str(round(W*96/25.4)),'pageHeight':str(round(H*96/25.4)),'data-coordinate-unit':'px'})
root=ET.SubElement(model,'root');ET.SubElement(root,'mxCell',{'id':'0'});ET.SubElement(root,'mxCell',{'id':'1','parent':'0'})
px=96/25.4
def mxrect(ident,x,y,w,h,fill,stroke):
    cell=ET.SubElement(root,'mxCell',{'id':ident,'value':'','style':f'rounded=1;arcSize=9;whiteSpace=wrap;html=0;fillColor={fill};strokeColor={stroke};strokeWidth=1;','vertex':'1','parent':'1'})
    ET.SubElement(cell,'mxGeometry',{'x':str(round(x*px)),'y':str(round(y*px)),'width':str(round(w*px)),'height':str(round(h*px)),'as':'geometry'})
def rect(ident,x,y,w,h,fill,stroke,linewidth=.3):
    se('rect',{'id':ident,'x':x,'y':y,'width':w,'height':h,'rx':1.4,'fill':fill,'stroke':stroke,'stroke-width':linewidth})
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=1.4',linewidth=linewidth*72/25.4,facecolor=fill,edgecolor=stroke))
    mxrect(ident,x,y,w,h,fill,stroke)

tid=0
def text(txt,x,y,size=font_pt,weight='normal',color='#243746',anchor='middle'):
    global tid
    tid+=1
    se('text',{'id':f'text-{tid}','x':x,'y':y,'text-anchor':anchor,'dominant-baseline':'central','font-family':font,'font-size':size*25.4/72,'font-weight':weight,'fill':color}).text=txt
    ax.text(x,y,txt,ha={'middle':'center','start':'left','end':'right'}[anchor],va='center',fontsize=size,fontfamily=font,fontweight=weight,color=color)
    cell=ET.SubElement(root,'mxCell',{'id':f'label-{tid}','value':txt,'style':f'text;html=0;align={"center" if anchor=="middle" else "left"};verticalAlign=middle;whiteSpace=wrap;fontFamily={font};fontSize={size*96/72};fontColor={color};fontStyle={1 if weight=="bold" else 0};strokeColor=none;fillColor=none;','vertex':'1','parent':'1'})
    width=84 if len(txt)>33 else 40
    left=x-width/2 if anchor=='middle' else x
    ET.SubElement(cell,'mxGeometry',{'x':str(round(left*px)),'y':str(round((y-3)*px)),'width':str(round(width*px)),'height':str(round(6*px)),'as':'geometry'})

main=spec['diagram']['nodes'][:5]
wrap_titles={
    'photograph':['RGB','photograph'],
    'segmentation':['Tissue','segmentation'],
    'mapping':['Image–array','mapping'],
    'rule':['Predefined','exposure rule'],
    'treatment':['Scheduled','treatment'],
}
for n in main:
    x,y,w,h=n['x'],n['y'],n['width_mm'],n['height_mm']
    accent=cfg['orange'] if n['id']=='rule' else cfg['blue']
    fill=cfg['pale_orange'] if n['id']=='rule' else cfg['pale_blue']
    rect(n['id'],x,y,w,h,fill,accent,.38)
    lines=wrap_titles[n['id']]
    text(lines[0],x+w/2,y+9,weight='bold')
    text(lines[1],x+w/2,y+13.3,weight='bold')
    text(n['label'].split('\n')[-1],x+w/2,y+23,cfg['subtitle_size_pt'],color='#5A6775')

byid={n['id']:n for n in main}
for i,e in enumerate(spec['diagram']['edges']):
    a,b=byid[e['source']],byid[e['target']]
    start=(a['x']+a['width_mm']+.6,a['y']+a['height_mm']/2)
    end=(b['x']-.9,b['y']+b['height_mm']/2)
    se('path',{'d':f'M {start[0]} {start[1]} L {end[0]} {end[1]}','stroke':'#687789','stroke-width':.45,'fill':'none','marker-end':'url(#arrow)'})
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=8,linewidth=.85,color='#687789',shrinkA=0,shrinkB=0))
    cell=ET.SubElement(root,'mxCell',{'id':f'edge-{i}','source':e['source'],'target':e['target'],'edge':'1','parent':'1','style':'edgeStyle=orthogonalEdgeStyle;rounded=0;html=0;endArrow=block;endFill=1;strokeColor=#687789;strokeWidth=1.5;exitX=1;exitY=0.5;entryX=0;entryY=0.5;'})
    ET.SubElement(cell,'mxGeometry',{'relative':'1','as':'geometry'})

rect('evaluation-strip',7,46,171,28,'#F7F9FC','#D4DFEB',.25)
text('Separately trained segmentation models',12,51.5,8.5,weight='bold',anchor='start',color='#415C77')
se('line',{'x1':92,'x2':92,'y1':55.5,'y2':70.5,'stroke':'#D4DFEB','stroke-width':.25})
ax.plot([92,92],[55.5,70.5],color='#D4DFEB',lw=.6)
text('Human DFUTissue',15,59.3,8.5,weight='bold',anchor='start')
text('Fibrin · granulation · callus',15,66,8.2,anchor='start')
text('Internal animal images',98,59.3,8.5,weight='bold',anchor='start')
text('Epithelial · granulation · necrotic · pus',98,66,8.2,anchor='start')

out=HERE.parents[1]/'figures';out.mkdir(exist_ok=True)
ET.ElementTree(svg).write(out/'Figure2A_workflow.svg',encoding='utf-8',xml_declaration=True)
ET.ElementTree(mx).write(HERE/'Figure2A_workflow.drawio',encoding='utf-8',xml_declaration=True)
fig.savefig(out/'Figure2A_workflow.pdf',dpi=600)
fig.savefig(out/'Figure2A_workflow.png',dpi=600)
fig.savefig(out/'Figure2A_workflow-preview.png',dpi=180)
plt.close(fig)
(HERE/'caption.txt').write_text(spec['figure']['caption']+'\n')
files=[HERE/'figure.yaml',HERE/'render_workflow.py',HERE/'Figure2A_workflow.drawio',*sorted(out.glob('Figure2A_workflow*'))]
manifest={'figure':'Figure2A_workflow','claim_anchor':'C2A','physical_size_mm':[W,H],'font_family':font,'minimum_font_pt':8.0,'raster_dpi':600,'preview_dpi':180,'editable_sources':['figure.yaml','render_workflow.py','Figure2A_workflow.drawio','../../figures/Figure2A_workflow.svg'],'files':{str(p.relative_to(HERE.parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'visual_review':{'status':'pending','passes':0}}
(HERE/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'exports':str(out),'size_mm':[W,H],'font':manifest['font_family']},indent=2))
