import pandas as pd
import json
import pydot
import networkx as nx

# DOT_FILE = "causal_graph.dot"

def generate_dot(n_obs, n_steps, DOT_FILE):
    dot_out = open(DOT_FILE, 'w')

    dot_out.writelines("digraph g {\n")

    for step in range(n_steps):
        if step < n_steps-1:
            dot_out.writelines("   \"A_" + str(step)+ "\" -> \"R_" + str(step) + "\" [arrowtail=none, arrowhead=normal]; \n")

        for obs in range(n_obs):
            # dot_out.writelines("   \"O" + str(obs) +"_"+ str(step)+ "\" -> \"R_" + str(step) + "\" [arrowtail=none, arrowhead=normal]; \n")
            if step < n_steps-1:
                # dot_out.writelines("   \"O" + str(obs) +"_"+ str(step)+ "\" -> \"R_" + str(step) + "\" [arrowtail=none, arrowhead=normal]; \n")
                dot_out.writelines("   \"A_" + str(step)+ "\" -> \"O" + str(obs) +"_"+ str(step+1)+ "\" [arrowtail=none, arrowhead=normal]; \n")
                dot_out.writelines("   \"O" + str(obs) +"_"+ str(step+1)+ "\" -> \"R_" + str(step) + "\" [arrowtail=none, arrowhead=normal]; \n")
                for obs_i in range(n_obs):
                        dot_out.writelines("   \"O" + str(obs) +"_"+ str(step)+ "\" -> \"O" + str(obs_i) +"_"+ str(step+1) + "\" [arrowtail=none, arrowhead=normal]; \n")

    dot_out.writelines("}")


def build_img(dot_file,img_file):
    dot_str = open(dot_file,'r').read()
    # print(dot_str)
    graphs = pydot.graph_from_dot_data(dot_str)
    svg_str = graphs[0].create_svg()
    svg_str = str(svg_str).replace("\\r\\n", "").replace("b\'","").replace("\'","").replace("ts&#45;","").replace("_f","")
    img_file = open(img_file, "w")
    img_file.write(svg_str)
    img_file.close()


def plan():
    return


# if __name__ == "__main__":
#     generate_dot(2, 4)
#     build_img(DOT_FILE, "causal_graph.svg")