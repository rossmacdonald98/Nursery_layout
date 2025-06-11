##### Code Summary #####

# This script defines an OpenMC model for the LIBRA BABY 1L experiment in the vaul, containing
# several BABY experiments with different breeder materials running in parallel. This model is to
# understand the amount of cross-irradiation that occurs with the neutron source in different positions.
# The position(s) and z-offset(s) of the neutron source can be specified by the user.
# The script will run through all combinations of source location and save the results.

# The code is strectured as follows:
# 1. Import necessary libraries and modules.
# 2. Define the vault layout, including BABY positions and breeder materials.
# 3. Define functions to calculate breeder depth,
#    Li2O bed properties, build the nursery model etc. Includes tally definitions in nursery_model().
# 5. Define the dimensions of the BABY experiments.
# 4. Define the materials for the BABY experiments.
# 5. Run the model in a loop to cycle through all source locations and z-offsets, save results in processed_data.json.

import os
import glob
import openmc
from libra_toolbox.neutronics.neutron_source import A325_generator_diamond
import vault_modified
import math
import numpy as np
import json

# Vault layout
## List of BABY coordinates within vault
baby_positions = [
    (590, 70, 100),
    (885, 78, 100),
    (897, 299, 100),
    # (700, 200, 100), # Uncomment to add a fourth BABY
]

# Breeder materials for each BABY experiment
# The order of the breeders should match the order of the BABY positions
breeders = ["ClLiF", "Li2O", "LiPb"]

## Source position
source_positions = [
    1,
    3,
]  # Indexes of the BABY position where the source is located, model runs for each position
source_z_offsets = [
    -5.635,
]  # Offsets for the source Z position below table (negative)

############################################################################
# Functions


def deep_update(d, u):
    """Recursively updates a dictionary with another dictionary.
    Args:
        d (dict): The dictionary to update.
        u (dict): The dictionary with updates.
    """
    for k, v in u.items():
        if isinstance(v, dict) and k in d:
            deep_update(d[k], v)
        else:
            d[k] = v


def calculate_breeder_depth(R, r, g, V):
    """Calculates the height (H) of a cylindrical volume (radius R & volume V) with an
    inserted inner cylinder (of radius r & gap from larger cylinder floor of g).

    Args:
        R (float): Major radius of breeder volume cylinder (cm)
        r (float): Radius of inner cylinder inserted into breeder volume (cm)
        g (float): Gap between floor of breeder volume cylinder and inserted inner cylinder (cm)
        V (float): Volume of breeder material(cm3)

    Returns:
        (float): Depth of breeder material volume.
    """
    v1 = math.pi * R**2 * g  # Volume of cylinder beneath heater
    v2 = V - v1  # Volume of annulus around heater

    h1 = g  # Height below heater
    h2 = v2 / (math.pi * (R**2 - r**2))  # Height of annulus around heater

    H = h1 + h2  # Total height for given volume
    return H


def get_Li2O_bed_properties(pellet_porosity, packing_efficiency, he_density):
    """Calculates Li2O pellet bed density and volumetric mass fractions of
    Li, O & He for a given pellet porosity, packing efficiency and Helium gas density

    Args:
        pellet_porosity (float): Porosity of Li2O pellets in pellet bed
        packing_efficiency (float): Volumetric packing efficiency of Li2O pellets in pellet bed
        he_density (float): Density of the helium gas in the pellet bed void space

        Returns:
        tuple: (Li_mass_frac_bed, O_mass_frac_bed, He_mass_frac_bed, pellet_bed_density)
    """

    Li_at_mass = 6.94  # Lithium molar mass
    O_at_mass = 16.0  # Oxygen molar mass

    Li2O_mol_mass = (2 * Li_at_mass) + O_at_mass  # molar mass of Li2O

    Li_mass_frac = (2 * Li_at_mass) / Li2O_mol_mass  # mass fraction of Lithium
    O_mass_frac = O_at_mass / Li2O_mol_mass  # mass fraction of Oxygen

    Li2O_density = 2.013  # Theoretical bulk density of Li2O, from 'Handbook of inorganic chemicals' by Pradyot Patnaik, Ph.D

    Li2O_pellet_density = (
        1 - pellet_porosity
    ) * Li2O_density  # Actual density of Li2O pellets

    Li2O_bed_density = (
        Li2O_pellet_density * packing_efficiency
    )  # Effective density of Li2O in pellet bed

    He_bed_density = he_density * (
        1 - packing_efficiency
    )  # Effective density of Helium in pellet bed. Helium density at 5psi & 300K

    pellet_bed_density = Li2O_bed_density + He_bed_density  # Density of the pellet bed

    Li_eff_density = (
        Li_mass_frac * Li2O_bed_density
    )  # Effective density of Lithium in pellet bed

    O_eff_density = (
        O_mass_frac * Li2O_bed_density
    )  # Effective density of Oxygen in pellet bed

    Li_mass_frac_bed = (
        Li_eff_density / pellet_bed_density
    )  # Mass fraction of Lithium per volume of packed pellet bed

    O_mass_frac_bed = (
        O_eff_density / pellet_bed_density
    )  # Mass fraction of Oxygen per volume of packed pellet bed

    He_mass_frac_bed = (
        He_bed_density / pellet_bed_density
    )  # Mass fraction of Helium per volume of packed pellet bed

    return pellet_bed_density, Li_mass_frac_bed, O_mass_frac_bed, He_mass_frac_bed


def nursery_model(src_position, src_z_offset):
    """Returns an openmc model of the 'nursery' vault containing several BABY experiments and returns a TBR for each of them.

    Returns:
        the openmc model
    """

    materials = [
        SS316L,
        Li2O_bed,
        lithium_lead,
        cllif_nat,
        SS304,
        heater_mat,
        furnace,
        alumina,
        lead,
        air,
        epoxy,
        he,
    ]

    ########## Build cells defining geometry of BABY experiment(s), source and exclusion sphere ##########
    # sphere = sphere_geometry(baby_positions)

    cells, breeder_cells = nursery_geometry(
        baby_positions, breeders, src_position, src_z_offset
    )

    source_x = baby_positions[src_position - 1][0]  # Get the x position of the source
    source_y = baby_positions[src_position - 1][1]  # Get the y position of the source
    source_z = (
        baby_positions[src_position - 1][2] + src_z_offset
    )  # Get the z position of the source

    ############################################################################
    # Define Settings

    settings = openmc.Settings()

    src = A325_generator_diamond((source_x, source_y, source_z), (1, 0, 0))
    settings.source = src
    settings.batches = 100
    settings.inactive = 0
    settings.run_mode = "fixed source"
    settings.particles = int(5e3)
    settings.output = {"tallies": True}
    settings.photon_transport = False

    ############################################################################
    # overall_exclusion_region = -sphere

    overall_exclusion_region = -bounding_geometry(baby_positions, 50)

    ############################################################################
    # Specify Tallies
    # Create a list of tallies with initial TBR tallies for each breeder cell
    tallies = openmc.Tallies()

    for i, breeder_cell in enumerate(breeder_cells, start=1):
        tally = openmc.Tally(name=f"TBR_{i}")
        tally.scores = ["(n,Xt)"]
        tally.filters = [openmc.CellFilter(breeder_cell)]
        tallies.append(tally)

    ############################################################################
    # Model

    model = vault_modified.build_vault_model(
        settings=settings,
        tallies=tallies,
        added_cells=cells,
        added_materials=materials,
        overall_exclusion_region=overall_exclusion_region,
    )

    # Get all cells from the model geometry including the vault geometry
    model_cells = model.geometry.get_all_cells()

    # Get wall cells from model geometry by filtering for concrete fill
    wall_cells = [cell for cell in model_cells.values() if getattr(cell.fill, "name", None) == "Concrete"]

    print(f"Found {len(wall_cells)} wall cells in the model.")

    # Get LiPb breeder cells from model geometry by filtering for lithium_lead fill
    LiPb_cell = [
        cell for cell in model_cells.values() if getattr(cell.fill, "name", None) == "Lithium Lead"
        ]

    # Create list for new tallies to add to model
    more_tallies = openmc.Tallies()

    # Create a tally for breeder TBR from neutrons from vault wall cells
    for i, breeder_cell in enumerate(breeder_cells, start=1):
        tally = openmc.Tally(name=f"TBR_from_wall_{i}")
        from_wall_cell_filter = openmc.CellFromFilter(wall_cells)
        tally.scores = ["(n,Xt)"]
        tally.filters = [openmc.CellFilter(breeder_cell),from_wall_cell_filter]
        more_tallies.append(tally)

        # Create a tally for breeder TBR from neutrons from vault wall cells
    for i, breeder_cell in enumerate(breeder_cells, start=1):
        tally = openmc.Tally(name=f"TBR_from_LiPb_{i}")
        LiPb_born_filter = openmc.CellBornFilter(LiPb_cell)
        tally.scores = ["(n,Xt)"]
        tally.filters = [openmc.CellFilter(breeder_cell),LiPb_born_filter]
        more_tallies.append(tally)

    # Get the energy spectrum of neutrons responsible for breeding tritium in the breeder cells
    for i, breeder_cell in enumerate(breeder_cells, start=1):
        energies = openmc.mgxs.GROUP_STRUCTURES["CCFE-709"]
        energy_filter = openmc.EnergyFilter(energies)
        tally = openmc.Tally(name=f"TBR_spectrum_{i}")
        tally.filters = [openmc.CellFilter(breeder_cell), energy_filter]
        tally.scores = ["(n,Xt)"]
        more_tallies.append(tally)

    model.tallies.extend(more_tallies)

    return model


def bounding_geometry(positions, margin):
    """
    Returns a cuboid region that bounds all positions, with optional margin.

    Args:
        positions (list of tuple): List of (x, y, z) positions.
        margin (float): Extra margin to add to each side (in cm).

    Returns:
        cuboid surface bounding all supplied positions with margin.
    """
    positions = np.array(positions)
    x_min, y_min, z_min = np.min(positions, axis=0) - margin
    x_max, y_max, z_max = np.max(positions, axis=0) + margin

    bouding_cuboid = openmc.model.RectangularParallelepiped(
        x_min, x_max, y_min, y_max, z_min, z_max
    )

    return bouding_cuboid


def nursery_geometry(baby_positions, breeders, src_position, src_z_offset):
    """Returns the geometry for the BABY experiments in the vault, with specified breeder materials and source location.

    Args:
        baby_positions: list of tuples defining all BABY positions in the vault (cm)
        breeders: list of strings defining the breeder material for each BABY experiment

    Returns:
        cells: cells defining the BABY geometries in their respective positions in the vault.
        breeder_cells: list of breeder cells for each BABY experiment
    """

    no_BABYs = len(baby_positions)

    cells_dict = {}
    breeder_cells_dict = {}
    trim_regions_dict = {}

    for i in range(no_BABYs):

        ########## BABY i ##########

        print(f"Building BABY {i+1} geometry...")

        x_c, y_c, z_c = baby_positions[i]

        breeder = breeders[i]

        ########## Surfaces ##########
        z_plane_1 = openmc.ZPlane(0.0 + z_c)
        z_plane_2 = openmc.ZPlane(epoxy_thickness + z_c)
        z_plane_3 = openmc.ZPlane(epoxy_thickness + alumina_compressed_thickness + z_c)
        z_plane_4 = openmc.ZPlane(
            epoxy_thickness + alumina_compressed_thickness + ov_base_thickness + z_c
        )
        z_plane_5 = openmc.ZPlane(
            epoxy_thickness
            + alumina_compressed_thickness
            + ov_base_thickness
            + alumina_thickness
            + z_c
        )
        z_plane_6 = openmc.ZPlane(
            epoxy_thickness
            + alumina_compressed_thickness
            + ov_base_thickness
            + alumina_thickness
            + he_thickness
            + z_c
        )
        z_plane_7 = openmc.ZPlane(
            epoxy_thickness
            + alumina_compressed_thickness
            + ov_base_thickness
            + alumina_thickness
            + he_thickness
            + iv_base_thickness
            + z_c
        )
        z_plane_8 = openmc.ZPlane(
            epoxy_thickness
            + alumina_compressed_thickness
            + ov_base_thickness
            + alumina_thickness
            + he_thickness
            + iv_base_thickness
            + breeder_thickness
            + z_c
        )
        z_plane_9 = openmc.ZPlane(
            epoxy_thickness
            + alumina_compressed_thickness
            + ov_base_thickness
            + alumina_thickness
            + he_thickness
            + iv_base_thickness
            + breeder_thickness
            + cover_he_thickness
            + z_c
        )
        z_plane_10 = openmc.ZPlane(
            epoxy_thickness
            + alumina_compressed_thickness
            + ov_base_thickness
            + alumina_thickness
            + he_thickness
            + iv_base_thickness
            + breeder_thickness
            + cover_he_thickness
            + iv_cap
            + z_c
        )
        z_plane_11 = openmc.ZPlane(
            epoxy_thickness
            + alumina_compressed_thickness
            + ov_base_thickness
            + alumina_thickness
            + furnace_thickness
            + z_c
        )
        z_plane_12 = openmc.ZPlane(
            epoxy_thickness
            + alumina_compressed_thickness
            + ov_base_thickness
            + ov_height
            + z_c
        )
        z_plane_13 = openmc.ZPlane(
            epoxy_thickness
            + alumina_compressed_thickness
            + ov_base_thickness
            + ov_height
            + ov_cap
            + z_c
        )
        z_plane_14 = openmc.ZPlane(z_c - table_height)
        z_plane_15 = openmc.ZPlane(z_c - table_height - epoxy_thickness)

        ########## Cylinders ##########
        z_cyl_1 = openmc.ZCylinder(x0=x_c, y0=y_c, r=breeder_radius)
        z_cyl_2 = openmc.ZCylinder(x0=x_c, y0=y_c, r=iv_external_radius)
        z_cyl_3 = openmc.ZCylinder(x0=x_c, y0=y_c, r=he_radius)
        z_cyl_4 = openmc.ZCylinder(x0=x_c, y0=y_c, r=furnace_radius)
        z_cyl_5 = openmc.ZCylinder(x0=x_c, y0=y_c, r=ov_internal_radius)
        z_cyl_6 = openmc.ZCylinder(x0=x_c, y0=y_c, r=ov_external_radius)

        heater_z = (
            epoxy_thickness
            + alumina_compressed_thickness
            + ov_base_thickness
            + alumina_thickness
            + he_thickness
            + iv_base_thickness
            + heater_gap
            + z_c
        )

        right_cyl = openmc.model.RightCircularCylinder(
            (x_c, y_c, heater_z), heater_length, heater_radius, axis="z"
        )

        if src_position == i + 1:
            # If BABY i is the one with the neutron source, add the source geometry
            source_x = x_c - 13.50
            source_y = y_c
            source_z = z_c + src_z_offset

            ext_cyl_source = openmc.model.RightCircularCylinder(
                (source_x, source_y, source_z), source_h, source_external_r, axis="x"
            )
            source_region = openmc.model.RightCircularCylinder(
                (source_x + 0.25, source_y, source_z),
                source_h - 0.50,
                source_internal_r,
                axis="x",
            )

            source_wall_region = -ext_cyl_source & +source_region
            source_region = -source_region

        ########## Cuboid for trimming geometry ##########

        x_min = x_c - 40
        x_max = x_c + 40
        y_min = y_c - 40
        y_max = y_c + 40
        z_min = z_c - 40
        z_max = z_c + 40

        cuboid = openmc.model.RectangularParallelepiped(
            x_min, x_max, y_min, y_max, z_min, z_max
        )
        trim_regions_dict[f"cuboid_{i+1}"] = cuboid

        ########## Lead bricks positioned under the source ##########
        positions = [
            (x_c - 13.50, y_c, z_c - table_height),
            (x_c - 4.50, y_c, z_c - table_height),
            (x_c + 36.50, y_c, z_c - table_height),
            (x_c + 27.50, y_c, z_c - table_height),
        ]

        lead_blocks = []
        for position in positions:
            lead_block_region = openmc.model.RectangularParallelepiped(
                position[0] - lead_width / 2,
                position[0] + lead_width / 2,
                position[1] - lead_length / 2,
                position[1] + lead_length / 2,
                position[2],
                position[2] + lead_height,
            )
            lead_blocks.append(lead_block_region)

        ########## Regions for BABY 1 ##########
        epoxy_region = +z_plane_1 & -z_plane_2 & -cuboid
        alumina_compressed_region = +z_plane_2 & -z_plane_3 & -cuboid
        bottom_vessel = +z_plane_3 & -z_plane_4 & -z_cyl_6
        top_vessel = +z_plane_12 & -z_plane_13 & -z_cyl_6 & +right_cyl
        cylinder_vessel = +z_plane_4 & -z_plane_12 & +z_cyl_5 & -z_cyl_6
        vessel_region = bottom_vessel | cylinder_vessel | top_vessel
        alumina_region = +z_plane_4 & -z_plane_5 & -z_cyl_5
        bottom_cap = +z_plane_6 & -z_plane_7 & -z_cyl_2 & +right_cyl
        cylinder_cap = +z_plane_7 & -z_plane_9 & +z_cyl_1 & -z_cyl_2 & +right_cyl
        top_cap = +z_plane_9 & -z_plane_10 & -z_cyl_2 & +right_cyl
        cap_region = bottom_cap | cylinder_cap | top_cap

        breeder_region = +z_plane_7 & -z_plane_8 & -z_cyl_1 & +right_cyl

        gap_region = +z_plane_8 & -z_plane_9 & -z_cyl_1 & +right_cyl
        furnace_region = +z_plane_5 & -z_plane_11 & +z_cyl_3 & -z_cyl_4
        heater_region = -right_cyl
        table_under_source_region = +z_plane_15 & -z_plane_14 & -cuboid
        lead_block_1_region = -lead_blocks[0]
        lead_block_2_region = -lead_blocks[1]
        lead_block_3_region = -lead_blocks[2]
        lead_block_4_region = -lead_blocks[3]

        if src_position == i + 1:
            he_region = (
                +z_plane_5
                & -z_plane_12
                & -z_cyl_5
                & ~source_region
                & ~epoxy_region
                & ~alumina_compressed_region
                & ~alumina_region
                & ~breeder_region
                & ~gap_region
                & ~furnace_region
                & ~vessel_region
                & ~cap_region
                & ~heater_region
                & ~table_under_source_region
                & ~lead_block_1_region
                & ~lead_block_2_region
                & ~lead_block_3_region
                & ~lead_block_4_region
            )
            cuboid_region = (
                -cuboid
                & ~source_wall_region
                & ~source_region
                & ~epoxy_region
                & ~alumina_compressed_region
                & ~alumina_region
                & ~breeder_region
                & ~gap_region
                & ~furnace_region
                & ~he_region
                & ~vessel_region
                & ~cap_region
                & ~heater_region
                & ~table_under_source_region
                & ~lead_block_1_region
                & ~lead_block_2_region
                & ~lead_block_3_region
                & ~lead_block_4_region
            )
        else:
            he_region = (
                +z_plane_5
                & -z_plane_12
                & -z_cyl_5
                & ~epoxy_region
                & ~alumina_compressed_region
                & ~alumina_region
                & ~breeder_region
                & ~gap_region
                & ~furnace_region
                & ~vessel_region
                & ~cap_region
                & ~heater_region
                & ~table_under_source_region
                & ~lead_block_1_region
                & ~lead_block_2_region
                & ~lead_block_3_region
                & ~lead_block_4_region
            )
            cuboid_region = (
                -cuboid
                & ~epoxy_region
                & ~alumina_compressed_region
                & ~alumina_region
                & ~breeder_region
                & ~gap_region
                & ~furnace_region
                & ~he_region
                & ~vessel_region
                & ~cap_region
                & ~heater_region
                & ~table_under_source_region
                & ~lead_block_1_region
                & ~lead_block_2_region
                & ~lead_block_3_region
                & ~lead_block_4_region
            )

        ########## Cells for BABY i ##########
        if src_position == i + 1:
            source_wall_cell = openmc.Cell(region=source_wall_region)
            source_wall_cell.fill = SS304
            cells_dict[f"source_wall_cell_{i+1}"] = source_wall_cell

            source_region = openmc.Cell(region=source_region)
            source_region.fill = None
            cells_dict[f"source_region_{i+1}"] = source_region

        epoxy_cell = openmc.Cell(region=epoxy_region)
        epoxy_cell.fill = epoxy
        cells_dict[f"epoxy_{i+1}"] = epoxy_cell

        alumina_compressed_cell = openmc.Cell(region=alumina_compressed_region)
        alumina_compressed_cell.fill = alumina
        cells_dict[f"alumina_compressed_{i+1}"] = alumina_compressed_cell

        vessel_cell = openmc.Cell(region=vessel_region)
        vessel_cell.fill = SS316L
        cells_dict[f"vessel_cell_{i+1}"] = vessel_cell

        alumina_cell = openmc.Cell(region=alumina_region)
        alumina_cell.fill = alumina
        cells_dict[f"alumina_cell_{i+1}"] = alumina_cell

        breeder_cell = openmc.Cell(region=breeder_region)
        if breeder == "Li2O":
            breeder_cell.fill = Li2O_bed
        elif breeder == "LiPb":
            breeder_cell.fill = lithium_lead
        elif breeder == "ClLiF":
            breeder_cell.fill = cllif_nat
        cells_dict[f"breeder_cell_{i+1}"] = breeder_cell
        breeder_cells_dict[f"breeder_cell_{i+1}"] = breeder_cell

        gap_cell = openmc.Cell(region=gap_region)
        gap_cell.fill = he
        cells_dict[f"gap_cell_{i+1}"] = gap_cell

        cap_cell = openmc.Cell(region=cap_region)
        cap_cell.fill = SS316L
        cells_dict[f"cap_cell_{i+1}"] = cap_cell

        furnace_cell = openmc.Cell(region=furnace_region)
        furnace_cell.fill = furnace
        cells_dict[f"furnace_cell_{i+1}"] = furnace_cell

        heater_cell = openmc.Cell(region=heater_region)
        heater_cell.fill = heater_mat
        cells_dict[f"heater_cell_{i+1}"] = heater_cell

        table_cell = openmc.Cell(region=table_under_source_region)
        table_cell.fill = epoxy
        cells_dict[f"table_cell_{i+1}"] = table_cell

        cuboid_cell = openmc.Cell(region=cuboid_region)
        cuboid_cell.fill = air
        cells_dict[f"cuboid_cell_{i+1}"] = cuboid_cell

        he_cell = openmc.Cell(region=he_region)
        he_cell.fill = he
        cells_dict[f"he_cell_{i+1}"] = he_cell

        lead_block_1_cell = openmc.Cell(region=lead_block_1_region)
        lead_block_1_cell.fill = lead
        cells_dict[f"lead_block_1_cell_{i+1}"] = lead_block_1_cell

        lead_block_2_cell = openmc.Cell(region=lead_block_2_region)
        lead_block_2_cell.fill = lead
        cells_dict[f"lead_block_2_cell_{i+1}"] = lead_block_2_cell

        lead_block_3_cell = openmc.Cell(region=lead_block_3_region)
        lead_block_3_cell.fill = lead
        cells_dict[f"lead_block_3_cell_{i+1}"] = lead_block_3_cell

        lead_block_4_cell = openmc.Cell(region=lead_block_4_region)
        lead_block_4_cell.fill = lead
        cells_dict[f"lead_block_4_cell_{i+1}"] = lead_block_4_cell

    ## Convert cells_dict to a list of cells
    cells = list(cells_dict.values())
    breeder_cells = list(breeder_cells_dict.values())
    trim_regions = list(trim_regions_dict.values())

    # extract cuboid region

    ########## Global sphere to enclose all BABY geometries ##########
    # global_sphere = sphere_geometry(baby_positions)

    global_cuboid = bounding_geometry(baby_positions, 50)

    outer_region = -global_cuboid
    for i in range(no_BABYs):
        cuboid_i = trim_regions[i]
        outer_region = outer_region & +cuboid_i

    outer_cell = openmc.Cell(region=outer_region)
    outer_cell.fill = air

    cells.append(outer_cell)

    return cells, breeder_cells


############################################################################
# Dimensions
# All dimensions in cm

## BABY vertical dimensions
epoxy_thickness = 2.54  # 1 inch
alumina_compressed_thickness = 2.54  # 1 inch
ov_base_thickness = 0.786
alumina_thickness = 0.635
he_thickness = 0.6
iv_base_thickness = 0.3
heater_gap = 0.878
iv_height = 10.8903
iv_cap = 1.422
furnace_thickness = 15.24
ov_height = 21.093
ov_cap = 2.392
table_height = 28.00
lead_height = 4.00
lead_width = 8.00
lead_length = 16.00

heater_length = 25.40

## BABY radial dimensions
heater_radius = 0.439
breeder_radius = 7.00
iv_external_radius = 7.3
he_radius = 9.144
furnace_radius = 14.224
ov_internal_radius = 17.561
ov_external_radius = 17.78

## Calculated dimensions
breeder_volume = 1000  # 1L = 1000 cm3

breeder_thickness = calculate_breeder_depth(
    breeder_radius, heater_radius, heater_gap, breeder_volume
)
cover_he_thickness = iv_height - breeder_thickness

## Source dimensions

source_h = 50.00
source_external_r = 5.00
source_internal_r = 4.75

############################################################################
# Define Materials
# Source: PNNL Materials Compendium April 2021
# PNNL-15870, Rev. 2

# 316L Stainless Steel
# Data from https://www.thyssenkrupp-materials.co.uk/stainless-steel-316l-14404.html
SS316L = openmc.Material(name="316L Steel")
SS316L.add_element("C", 0.0003, "wo")
SS316L.add_element("Si", 0.01, "wo")
SS316L.add_element("Mn", 0.02, "wo")
SS316L.add_element("P", 0.00045, "wo")
SS316L.add_element("S", 0.000151, "wo")
SS316L.add_element("Cr", 0.175, "wo")
SS316L.add_element("Ni", 0.115, "wo")
SS316L.add_element("N", 0.001, "wo")
SS316L.add_element("Mo", 0.00225, "wo")
SS316L.add_element("Fe", 0.655599, "wo")

SS316L.set_density("g/cm3", 8)

# helium @5psig
he_pressure = 34473.8  # Pa ~ 5 psig
he_temperature = 300  # K
R_he = 2077  # J/(kg*K)
he_density = he_pressure / (R_he * he_temperature) / 1000  # in g/cm^3
he = openmc.Material(name="Helium")
he.add_element("He", 1.0, "ao")
he.set_density("g/cm3", he_density)

# Lithium Oxide Pebble Bed
Li2O_bed = openmc.Material(name="Lithium Oxide Pebble Bed")
pellet_porosity = 0.05  # Current value a guess, data not available.
packing_efficiency = 0.7  # Random packing efficiency for cylindrical pellets with an aspect ratio of 1 **Needs citation**

pellet_bed_density, Li_mass_frac_bed, O_mass_frac_bed, He_mass_frac_bed = (
    get_Li2O_bed_properties(pellet_porosity, packing_efficiency, he_density)
)

Li2O_bed.add_element("O", O_mass_frac_bed, "wo")
Li2O_bed.add_element("Li", Li_mass_frac_bed, "wo")
Li2O_bed.add_element("He", He_mass_frac_bed, "wo")
Li2O_bed.set_density("g/cm3", pellet_bed_density)

# Lithium-Lead
# Composition from certificate of analysis provided with Lithium-Lead from Camex
lithium_lead = openmc.Material(name="Lithium Lead")
lithium_lead.add_element("Pb", 0.993479, "wo")
lithium_lead.add_element("Li", 0.0064, "wo")
lithium_lead.add_element("Tl", 0.00002, "wo")
lithium_lead.add_element("Zn", 0.000002, "wo")
lithium_lead.add_element("Sn", 0.000002, "wo")
lithium_lead.add_element("Sb", 0.000002, "wo")
lithium_lead.add_element("Ni", 0.000001, "wo")
lithium_lead.add_element("Cu", 0.000002, "wo")
lithium_lead.add_element("Cd", 0.000002, "wo")
lithium_lead.add_element("Bi", 0.00008, "wo")
lithium_lead.add_element("As", 0.000002, "wo")
lithium_lead.add_element("Ag", 0.000008, "wo")
lithium_lead.set_density("g/cm3", 9.10411395)  # Density at 600C

# lif-licl - natural - pure
licl_frac = 0.695
cllif_nat = openmc.Material(name="ClLiF natural")
cllif_nat.add_element("F", 0.5 * (1 - licl_frac), "ao")
cllif_nat.add_element("Li", 0.5 * (1 - licl_frac) + 0.5 * licl_frac, "ao")
cllif_nat.add_element("Cl", 0.5 * licl_frac, "ao")
cllif_nat.set_density("g/cm3", 2.1)  # Density at 600C

# Stainless Steel 304 from PNNL Materials Compendium (PNNL-15870 Rev2)
SS304 = openmc.Material(name="Stainless Steel 304")
# SS304.temperature = 700 + 273
SS304.add_element("C", 0.000800, "wo")
SS304.add_element("Mn", 0.020000, "wo")
SS304.add_element("P", 0.000450, "wo")
SS304.add_element("S", 0.000300, "wo")
SS304.add_element("Si", 0.010000, "wo")
SS304.add_element("Cr", 0.190000, "wo")
SS304.add_element("Ni", 0.095000, "wo")
SS304.add_element("Fe", 0.683450, "wo")
SS304.set_density("g/cm3", 8.00)

# Heater
heater_mat = openmc.Material(name="heater")
heater_mat.add_element("C", 0.000990, "wo")
heater_mat.add_element("Al", 0.003960, "wo")
heater_mat.add_element("Si", 0.004950, "wo")
heater_mat.add_element("P", 0.000148, "wo")
heater_mat.add_element("S", 0.000148, "wo")
heater_mat.add_element("Ti", 0.003960, "wo")
heater_mat.add_element("Cr", 0.215000, "wo")
heater_mat.add_element("Mn", 0.004950, "wo")
heater_mat.add_element("Fe", 0.049495, "wo")
heater_mat.add_element("Co", 0.009899, "wo")
heater_mat.add_element("Ni", 0.580000, "wo")
heater_mat.add_element("Nb", 0.036500, "wo")
heater_mat.add_element("Mo", 0.090000, "wo")
heater_mat.set_density("g/cm3", 2.44)

# Using Microtherm with 1 a% Al2O3, 27 a% ZrO2, and 72 a% SiO2
# https://www.foundryservice.com/product/microporous-silica-insulating-boards-mintherm-microtherm-1925of-grades/
furnace = openmc.Material(name="Furnace")
# Estimate average temperature of Firebrick to be around 300 C
# Firebrick.temperature = 273 + 300
furnace.add_element("Al", 0.004, "ao")
furnace.add_element("O", 0.666, "ao")
furnace.add_element("Si", 0.240, "ao")
furnace.add_element("Zr", 0.090, "ao")
furnace.set_density("g/cm3", 0.30)

# alumina insulation
# data from https://precision-ceramics.com/materials/alumina/
alumina = openmc.Material(name="Alumina insulation")
alumina.add_element("O", 0.6, "ao")
alumina.add_element("Al", 0.4, "ao")
alumina.set_density("g/cm3", 3.98)

# air
air = openmc.Material(name="Air")
air.add_element("C", 0.00012399, "wo")
air.add_element("N", 0.75527, "wo")
air.add_element("O", 0.23178, "wo")
air.add_element("Ar", 0.012827, "wo")
air.set_density("g/cm3", 0.0012)

# epoxy
epoxy = openmc.Material(name="Epoxy")
epoxy.add_element("C", 0.70, "wo")
epoxy.add_element("H", 0.08, "wo")
epoxy.add_element("O", 0.15, "wo")
epoxy.add_element("N", 0.07, "wo")
epoxy.set_density("g/cm3", 1.2)

# lead
# data from https://wwwrcamnl.wr.usgs.gov/isoig/period/pb_iig.html
lead = openmc.Material()
lead.set_density("g/cm3", 11.34)
lead.add_nuclide("Pb204", 0.014, "ao")
lead.add_nuclide("Pb206", 0.241, "ao")
lead.add_nuclide("Pb207", 0.221, "ao")
lead.add_nuclide("Pb208", 0.524, "ao")

############################################################################
# Main

# Create results directory.
processed_data = {}

if __name__ == "__main__":

    # Determine number of openmc runs based on source positions and z-offsets
    no_runs = len(source_positions) * len(source_z_offsets)

    # Initialize run counter
    run = 1

    # Delete any existing statepoint and summary files
    for file in glob.glob("*.h5"):
        os.remove(file)

    # Delete any existing processed_data.json file
    processed_data_file = "../../data/processed_data.json"
    if os.path.exists(processed_data_file):
        os.remove(processed_data_file)

    # For each source position...
    for src_position in source_positions:
        src_position_key = f"source position {src_position}"
        processed_data[src_position_key] = {}

        # For each z_offset, run the nursery model and save results into processed_data
        for src_z_offset in source_z_offsets:
            src_z_offset_key = f"z_offset {src_z_offset:+.3f}"
            processed_data[src_position_key][src_z_offset_key] = {}

            print(
                "Running nursery model for source position "
                f"{src_position} and z-offset {src_z_offset:.3f} cm..."
            )

            model = nursery_model(src_position, src_z_offset)
            model.run()

            # Load the statepoint file to extract results
            sp = openmc.StatePoint(f"statepoint.{model.settings.batches}.h5")

            # Print run results summary

            print(
                f" Nursery model simulation {run} of {no_runs} completed successfully."
            )
            print("BABY locations and breeder materials:")

            for i, (pos, breeder) in enumerate(zip(baby_positions, breeders), start=1):
                print(
                    f"  BABY {i} at position {pos} cm with breeder material: {breeder}"
                )

            print(f"Source position: {src_position}, Source z-offset: {src_z_offset}")

            # Extract and print TBR results for each BABY experiment for this source position and z-offset
            for i in range(len(baby_positions)):
                # Standard TBR tally
                tally_name = f"TBR_{i+1}"
                tbr_tally = sp.get_tally(name=tally_name).get_pandas_dataframe()

                mean = tbr_tally["mean"].iloc[0]
                stdev = tbr_tally["std. dev."].iloc[0]
                rel_stdev = stdev / mean

                print(f"BABY {i+1} TBR: {mean:.6e}\n")
                print(f"BABY {i+1} TBR std. dev: {stdev:.6e}\n")
                print(f"BABY {i+1} Relative standard deviation: {rel_stdev:.6e}\n")
                print("Relative standard deviation below 1e-02 (1%) indicates good convergence.")

                # Store standard TBR results
                processed_data[src_position_key][src_z_offset_key][f"modelled_TBR_{i+1}"] = {
                    "mean": mean,
                    "std_dev": stdev,
                    "relative_std_dev": rel_stdev,
                }

                # TBR from wall tally
                wall_tally_name = f"TBR_from_wall_{i+1}"
                tbr_wall_tally = sp.get_tally(name=wall_tally_name).get_pandas_dataframe()

                wall_mean = tbr_wall_tally["mean"].iloc[0]
                wall_stdev = tbr_wall_tally["std. dev."].iloc[0]
                wall_rel_stdev = wall_stdev / wall_mean

                print(f"BABY {i+1} TBR from wall: {wall_mean:.6e}\n")
                print(f"BABY {i+1} TBR from wall std. dev: {wall_stdev:.6e}\n")
                print(f"BABY {i+1} TBR from wall Relative std. dev: {wall_rel_stdev:.6e}\n")

                # Store wall TBR results
                processed_data[src_position_key][src_z_offset_key][f"modelled_TBR_from_wall_{i+1}"] = {
                    "mean": wall_mean,
                    "std_dev": wall_stdev,
                    "relative_std_dev": wall_rel_stdev,
                }

                # TBR from LiPb tally
                LiPb_tally_name = f"TBR_from_LiPb_{i+1}"
                tbr_LiPb_tally = sp.get_tally(name=LiPb_tally_name).get_pandas_dataframe()

                LiPb_mean = tbr_LiPb_tally["mean"].iloc[0]
                LiPb_stdev = tbr_LiPb_tally["std. dev."].iloc[0]
                LiPb_rel_stdev = LiPb_stdev / LiPb_mean

                print(f"BABY {i+1} TBR from LiPb: {LiPb_mean:.6e}\n")
                print(f"BABY {i+1} TBR from LiPb std. dev: {LiPb_stdev:.6e}\n")
                print(f"BABY {i+1} TBR from LiPb Relative std. dev: {LiPb_rel_stdev:.6e}\n")

                # Store LiPb TBR results
                processed_data[src_position_key][src_z_offset_key][f"modelled_TBR_from_LiPb_{i+1}"] = {
                    "mean": LiPb_mean,
                    "std_dev": LiPb_stdev,
                    "relative_std_dev": LiPb_rel_stdev,
                }

                # # Energy spectrum tally
                # spectrum_tally_name = f"TBR_spectrum_{i+1}"
                # spectrum_tally = sp.get_tally(name=spectrum_tally_name).get_pandas_dataframe()

                # print(spectrum_tally)

                # spectrum_mean = spectrum_tally["mean"].iloc[0]
                # spectrum_stdev = spectrum_tally["std. dev."].iloc[0]
                # spectrum_rel_stdev = spectrum_stdev / spectrum_mean

                # print(f"BABY {i+1} TBR energy spectrum: {spectrum_mean:.6e}\n")
                # print(f"BABY {i+1} TBR energy spectrum std. dev: {spectrum_stdev:.6e}\n")
                # print(f"BABY {i+1} TBR energy spectrum Relative std. dev: {spectrum_rel_stdev:.6e}\n")

                # # Store energy spectrum TBR results
                # processed_data[src_position_key][src_z_offset_key][f"modelled_TBR_spectrum_{i+1}"] = {
                #     "mean": spectrum_mean,
                #     "std_dev": spectrum_stdev,
                #     "relative_std_dev": spectrum_rel_stdev,
                # }

            # rename summary.h5 and statepoint files to enable next run
            # Rename summary.h5
            if os.path.exists("summary.h5"):
                os.rename("summary.h5", f"summary.{src_position}.{src_z_offset}.h5")

            # Rename statepoint file
            statepoint_file = f"statepoint.{model.settings.batches}.h5"
            new_statepoint_file = (
                f"statepoint.{src_position}.{src_z_offset}.{model.settings.batches}.h5"
            )
            if os.path.exists(statepoint_file):
                os.rename(statepoint_file, new_statepoint_file)

            processed_data_file = "../../data/processed_data.json"

            try:
                with open(processed_data_file, "r") as f:
                    existing_data = json.load(f)
            except FileNotFoundError:
                print(
                    f"Processed data file not found, creating it in {processed_data_file}"
                )
                existing_data = {}

            deep_update(existing_data, processed_data)

            with open(processed_data_file, "w") as f:
                json.dump(existing_data, f, indent=4)

            print(f"Processed data stored in {processed_data_file}")

            run = run + 1
print("All runs completed successfully.")
