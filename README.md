to run and render
```bash
python hopkarp.py > size4.graph.anim
manim -pql manimate.py BipartiteGraphAnimation
```

the graph input selection is hard coded in both hopkarp.py and manimate.py   
shameful, but temporary  

## Requirements
* manim
* latex

*to run without latex, set `labels=False` in the graphs in manimate.py 