to run and render
```bash
python hopkarp.py > size4.graph.anim
manim -plq manim_test.py BipartiteGraphAnimation
```

the graph input selection is hard coded in both hopkarp.py and manim_test.py   
shameful, but temporary  

## Requirements
* manim
* latex

*to run without latex, set `labels=False` in the graphs in manim_test.py 