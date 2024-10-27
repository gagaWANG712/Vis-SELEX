import sys
import argparse
import vermouth
import vermouth.forcefield

def get_options():
    parser = argparse.ArgumentParser(
        description='A code to generate .bld file according to full atoms itp'
        )
    parser.add_argument('-f', '--file', type=str, default='dna.gro', 
                        help='A file of full atoms gro'
    )
    parser.add_argument('-itp', '--itp', type=str, default='full.itp', 
                        help="A file of full atoms itp")
    
    if len(sys.argv) < 5:
        parser.print_help()
        parser.exit('Missing input options')
        
    return parser.parse_args()

def build_bld():
    opts = get_options()

    ignore_resnames=['TIP3']
    ff = vermouth.forcefield.ForceField("all_atom")
    with open(opts.itp, "r") as itpfile:
        lines = itpfile.readlines()
    vermouth.gmx.read_itp(lines, ff)

    system = vermouth.System()
    vermouth.GROInput(str(opts.file),
                    exclude=ignore_resnames,
                    ignh=False
                    ).run_system(system)

    mol_w_coords = system.molecules[0]
    mol_w_itp = list(ff.blocks.values())[0]
    mol_w_itp.make_edges_from_interaction_type('bonds')
    mol_res_graph = vermouth.graph_utils.make_residue_graph(mol_w_itp)

    with open("full_build.bld", "w") as _file:
        had_residues = []
        for res in mol_res_graph.nodes:
            resname = mol_res_graph.nodes[res]['resname']
            if resname in had_residues:
                continue
            had_residues.append(resname)
            _file.write("[ template ]\n")
            _file.write(f"resname {resname}\n")
            _file.write(f"[ atoms ]\n")
            res_graph = mol_res_graph.nodes[res]['graph']
            for node in res_graph.nodes:
                name = res_graph.nodes[node]['atomname']
                atype = res_graph.nodes[node]['atype']
                pos = mol_w_coords.nodes[node]['position']
                x = str(pos[0])
                y = str(pos[1])
                z = str(pos[2])
                _file.write(f"{name} {atype} {x} {y} {z}\n")
            _file.write("[ bonds ]\n")
            for a, b in res_graph.edges:
                nameA = res_graph.nodes[a]['atomname']
                nameB = res_graph.nodes[b]['atomname']
                _file.write(f"{nameA} {nameB}\n")

    print('Finished!')

if __name__ == '__main__':
    build_bld()
