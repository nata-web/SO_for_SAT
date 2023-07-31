import numpy as np
import json,textwrap
import shapely, shapely.affinity
from pyproj import Geod

geod = Geod(ellps="WGS84")

def loadShapes(dataset):
    
    with open("./" + dataset, encoding="utf8") as f:
        data = json.load(f)
    
    if dataset == 'CShapes-2.0.geojson':
      countries = {c['properties']['cntry_name']:shapely.geometry.shape(c['geometry']) for c in data['features'] if c['properties']["gwedate"]== "30.12.2019 23:00:00"}
    elif dataset == 'gadm41_JPN_1.json':
      countries = {c['properties']['NAME_1']:shapely.geometry.shape(c['geometry']) for c in data['features']}
    return countries

def calcBorders(countries):
  borders=[]
  for ind1,(c1,s1) in enumerate(countries.items()):
    for ind2,(c2,s2) in enumerate(countries.items()):
      if c1!=c2:
        try:
          intersect=shapely.intersection(s1,s2)
          intersect=geod.geometry_length(intersect)
        except:
          intersect=False
          print('FAILED',c1,c2)
        
        if intersect:
          borders.append([ind1,c1,ind2,c2,intersect]) 
  return borders

def toSvg(outName, countries, colorIndices = None, colors=['#abcdef','#123456']):
  if colorIndices is None:
    colorIndices = range(len(countries))
    
  with open(outName, 'w') as f:
    #specify margin in coordinate units
    margin = 5

    bbox=None
    shapes=[]
    for s in countries.values():
      flipped=shapely.affinity.scale(s,yfact=-1,origin=(0,0))
      shapes.append(flipped)

      thisBox=flipped.bounds
      if bbox is None:
        bbox = list(thisBox)
      else:
        bbox = [min(thisBox[0],bbox[0]), min(thisBox[1],bbox[1]),
                max(thisBox[2],bbox[2]), max(thisBox[3],bbox[3])]
    bbox[0] -= margin
    bbox[1] -= margin
    bbox[2] += margin
    bbox[3] += margin

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    #transform each coordinate unit into "scale" pixels
    scale = 10 # 2

    props = {
        'version': '1.1',
        'baseProfile': 'full',
        'width': '{width:.0f}px'.format(width = width*scale),
        'height': '{height:.0f}px'.format(height = height*scale),
        'viewBox': '%.1f,%.1f,%.1f,%.1f' % (bbox[0], bbox[1], width, height),
        'xmlns': 'http://www.w3.org/2000/svg',
        'xmlns:ev': 'http://www.w3.org/2001/xml-events',
        'xmlns:xlink': 'http://www.w3.org/1999/xlink',
    }

    f.write(textwrap.dedent(r'''<?xml version="1.0" encoding="utf-8" ?>
        <svg {attrs:s}>
    ''').format(
        attrs = ' '.join(['{key:s}="{val:s}"'.format(key = key, val = props[key]) for key in props])))
    for s,cInd in zip(shapes,colorIndices):
      f.write(s.svg(scale_factor = 0.1, fill_color = colors[cInd%len(colors)])+'\n')
    f.write('''</svg>''')   


def makeCheckerboard(n):
    sc=10;
    Checker = {chr(ord('a')+a)+chr(ord('1')+b):shapely.box(a*sc,b*sc,(a+1)*sc,(b+1)*sc) for a in range(n) for b in range(n)} 
    return Checker

countries = loadShapes('CShapes-2.0.geojson')

europeBbox = shapely.box(countries['Spain'].bounds[0],countries['Spain'].bounds[1],countries['Belarus (Byelorussia)'].bounds[2],countries['Norway'].bounds[3])
Europe = {k:v for k,v in countries.items() if europeBbox.buffer(1).contains(v) and k!='Tunisia'}

SouthAmericaBbox = shapely.box(countries['Chile'].bounds[0],countries['Chile'].bounds[1],countries['Brazil'].bounds[2],countries['Venezuela'].bounds[3])
SouthAmerica = {k:v for k,v in countries.items() if SouthAmericaBbox.buffer(1).contains(v)}

Japan = loadShapes('gadm41_JPN_1.json')

# JapanBbox = shapely.box(Japan['Kagoshima'].bounds[0], Japan['Kagoshima'].bounds[1], 
#                         Japan['Hokkaido'].bounds[2], Japan['Hokkaido'].bounds[3])
# JapanCut = {k:v for k,v in prefJapan.items() if JapanBbox.buffer(1).contains(v)}
             
            

