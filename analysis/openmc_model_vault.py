import openmc
from libra_toolbox.neutronics.neutron_source import A325_generator_diamond
from libra_toolbox.neutronics import vault
import math
import numpy as np

############################################################################
# Functions


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


def nursery_model():
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
    sphere = sphere_geometry(baby_positions)
    
    cells, breeder_cells = nursery_geometry(baby_positions, breeders)

    ############################################################################
    # Define Settings

    settings = openmc.Settings()

    src = A325_generator_diamond((source_x, source_y, source_z), (1, 0, 0))
    settings.source = src
    settings.batches = 100
    settings.inactive = 0
    settings.run_mode = "fixed source"
    settings.particles = int(1e4)
    settings.output = {"tallies": True}
    settings.photon_transport = False

    ############################################################################
    overall_exclusion_region = -sphere

    ############################################################################
    # Specify Tallies

    # Extract breeder cells for tallies
    breeder_1 = breeder_cells[0]
    breeder_2 = breeder_cells[1]
    breeder_3 = breeder_cells[2]

    # Create a list of tallies
    tallies = openmc.Tallies()

    # Create tally for Li2O cell 1 TBR results
    tbr_tally_1 = openmc.Tally(name="TBR_1")
    tbr_tally_1.scores = ["(n,Xt)"]
    tbr_tally_1.filters = [openmc.CellFilter(breeder_1)]  # Add cell filter to tally

    tallies.append(tbr_tally_1)

    # Create tally for Li2O cell 2 TBR results
    tbr_tally_2 = openmc.Tally(name="TBR_2")
    tbr_tally_2.scores = ["(n,Xt)"]
    tbr_tally_2.filters = [openmc.CellFilter(breeder_2)]  # Add cell filter to tally

    tallies.append(tbr_tally_2)

    # Create tally for Li2O cell 3 TBR results
    tbr_tally_3 = openmc.Tally(name="TBR_3")
    tbr_tally_3.scores = ["(n,Xt)"]
    tbr_tally_3.filters = [openmc.CellFilter(breeder_3)]  # Add cell filter to tally

    tallies.append(tbr_tally_3)

    ############################################################################
    # Model

    model = vault.build_vault_model(
        settings=settings,
        tallies=tallies,
        added_cells=cells,
        added_materials=materials,
        overall_exclusion_region=overall_exclusion_region,
    )

    return model


def sphere_geometry(baby_positions):
    """Returns the geometry for the model exclusion sphere. sphere is sized so that all BABY positions are included + 20%

    Args:
        baby_positions: list of tuples defining all BABY positions in the vault (cm)

    Returns:
        exclusion sphere for the OpenMC model.
    """

    # Calculate midpoint position (center of bounding box)
    min_xyz = np.min(baby_positions, axis=0)
    max_xyz = np.max(baby_positions, axis=0)
    midpoint = (min_xyz + max_xyz) / 2
    x_c, y_c, z_c = midpoint

    # Calculate distances from midpoint to each position
    distances = [np.linalg.norm(np.array([x_c, y_c, z_c]) - np.array(pos)) for pos in baby_positions]

    # Find the maximum distance
    max_distance = max(distances)
    # Add 20% to the maximum distance for the sphere radius
    sphere_radius = max_distance * 1.2

    ########## Sphere ##########
    sphere = openmc.Sphere(x0=x_c, y0=y_c, z0=z_c, r=sphere_radius)  # before r=50.00

    return sphere


def nursery_geometry(baby_positions, breeders):
    """Returns the geometry for the BABY experiments in the vault, with specified breeder materials and source location.

    Args:
        baby_positions: list of tuples defining all BABY positions in the vault (cm)
        breeders: list of strings defining the breeder material for each BABY experiment

    Returns:
        cells: cells defining the BABY geometries in their respective positions in the vault.
        breeder_cells: list of breeder cells for each BABY experiment
    """


    ########## BABY 1 ##########
    x_c, y_c, z_c = baby_positions[0]

    breeder_1 = breeders[0]

    ########## Surfaces ##########
    z_plane_1_1 = openmc.ZPlane(0.0 + z_c)
    z_plane_2_1 = openmc.ZPlane(epoxy_thickness + z_c)
    z_plane_3_1 = openmc.ZPlane(epoxy_thickness + alumina_compressed_thickness + z_c)
    z_plane_4_1 = openmc.ZPlane(
        epoxy_thickness + alumina_compressed_thickness + ov_base_thickness + z_c
    )
    z_plane_5_1 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + alumina_thickness
        + z_c
    )
    z_plane_6_1 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + alumina_thickness
        + he_thickness
        + z_c
    )
    z_plane_7_1 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + alumina_thickness
        + he_thickness
        + iv_base_thickness
        + z_c
    )
    z_plane_8_1 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + alumina_thickness
        + he_thickness
        + iv_base_thickness
        + breeder_thickness
        + z_c
    )
    z_plane_9_1 = openmc.ZPlane(
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
    z_plane_10_1 = openmc.ZPlane(
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
    z_plane_11_1 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + alumina_thickness
        + furnace_thickness
        + z_c
    )
    z_plane_12_1 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + ov_height
        + z_c
    )
    z_plane_13_1 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + ov_height
        + ov_cap
        + z_c
    )
    z_plane_14_1 = openmc.ZPlane(z_c - table_height)
    z_plane_15_1 = openmc.ZPlane(z_c - table_height - epoxy_thickness)

    ########## Cylinders ##########
    z_cyl_1_1 = openmc.ZCylinder(x0=x_c, y0=y_c, r=breeder_radius)
    z_cyl_2_1 = openmc.ZCylinder(x0=x_c, y0=y_c, r=iv_external_radius)
    z_cyl_3_1 = openmc.ZCylinder(x0=x_c, y0=y_c, r=he_radius)
    z_cyl_4_1 = openmc.ZCylinder(x0=x_c, y0=y_c, r=furnace_radius)
    z_cyl_5_1 = openmc.ZCylinder(x0=x_c, y0=y_c, r=ov_internal_radius)
    z_cyl_6_1 = openmc.ZCylinder(x0=x_c, y0=y_c, r=ov_external_radius)

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

    right_cyl_1 = openmc.model.RightCircularCylinder(
        (x_c, y_c, heater_z), heater_length, heater_radius, axis="z"
    )

    if source_position == 1:
        # If BABY 1 is the one with the neutron source, add the source geometry
        source_x = x_c - 13.50
        source_y = y_c
        source_z = z_c - source_z_offset 

        ext_cyl_source = openmc.model.RightCircularCylinder(
        (source_x, source_y, source_z), source_h, source_external_r, axis="x"
        )
        source_region = openmc.model.RightCircularCylinder(
        (source_x + 0.25, source_y, source_z), source_h - 0.50, source_internal_r, axis="x"
        )

        source_wall_region = -ext_cyl_source & +source_region
        source_region = -source_region


    ########## Sphere for trimming geometry ##########
    sphere_1 = openmc.Sphere(x0=x_c, y0=y_c, z0=z_c, r=50.00)  

    ########## Lead bricks positioned under the source ##########
    positions = [
        (x_c - 13.50, y_c, z_c - table_height),
        (x_c - 4.50, y_c, z_c - table_height),
        (x_c + 36.50, y_c, z_c - table_height),
        (x_c + 27.50, y_c, z_c - table_height),
    ]

    lead_blocks_1 = []
    for position in positions:
        lead_block_region_1 = openmc.model.RectangularParallelepiped(
            position[0] - lead_width / 2,
            position[0] + lead_width / 2,
            position[1] - lead_length / 2,
            position[1] + lead_length / 2,
            position[2],
            position[2] + lead_height,
        )
        lead_blocks_1.append(lead_block_region_1)

    ########## Regions for BABY 1 ##########
    epoxy_region_1 = +z_plane_1_1 & -z_plane_2_1 & -sphere_1
    alumina_compressed_region_1 = +z_plane_2_1 & -z_plane_3_1 & -sphere_1
    bottom_vessel_1 = +z_plane_3_1 & -z_plane_4_1 & -z_cyl_6_1
    top_vessel_1 = +z_plane_12_1 & -z_plane_13_1 & -z_cyl_6_1 & +right_cyl_1
    cylinder_vessel_1 = +z_plane_4_1 & -z_plane_12_1 & +z_cyl_5_1 & -z_cyl_6_1
    vessel_region_1 = bottom_vessel_1 | cylinder_vessel_1 | top_vessel_1
    alumina_region_1 = +z_plane_4_1 & -z_plane_5_1 & -z_cyl_5_1
    bottom_cap_1 = +z_plane_6_1 & -z_plane_7_1 & -z_cyl_2_1 & +right_cyl_1
    cylinder_cap_1 = +z_plane_7_1 & -z_plane_9_1 & +z_cyl_1_1 & -z_cyl_2_1 & +right_cyl_1
    top_cap_1 = +z_plane_9_1 & -z_plane_10_1 & -z_cyl_2_1 & +right_cyl_1
    cap_region_1 = bottom_cap_1 | cylinder_cap_1 | top_cap_1

    breeder_region_1 = (
        +z_plane_7_1
        & -z_plane_8_1
        & -z_cyl_1_1
        & +right_cyl_1
    )

    gap_region_1 = +z_plane_8_1 & -z_plane_9_1 & -z_cyl_1_1 & +right_cyl_1
    furnace_region_1 = +z_plane_5_1 & -z_plane_11_1 & +z_cyl_3_1 & -z_cyl_4_1
    heater_region_1 = -right_cyl_1
    table_under_source_region_1 = +z_plane_15_1 & -z_plane_14_1 & -sphere_1
    lead_block_1_region_1 = -lead_blocks_1[0]
    lead_block_2_region_1 = -lead_blocks_1[1]
    lead_block_3_region_1 = -lead_blocks_1[2]
    lead_block_4_region_1 = -lead_blocks_1[3]

    if source_position == 1:
        he_region_1 = (
            +z_plane_5_1
            & -z_plane_12_1
            & -z_cyl_5_1
            & ~source_region
            & ~epoxy_region_1
            & ~alumina_compressed_region_1
            & ~alumina_region_1
            & ~breeder_region_1
            & ~gap_region_1
            & ~furnace_region_1
            & ~vessel_region_1
            & ~cap_region_1
            & ~heater_region_1
            & ~table_under_source_region_1
            & ~lead_block_1_region_1
            & ~lead_block_2_region_1
            & ~lead_block_3_region_1
            & ~lead_block_4_region_1
            )
        sphere_region_1 = (
            -sphere_1
            & ~source_wall_region
            & ~source_region
            & ~epoxy_region_1
            & ~alumina_compressed_region_1
            & ~alumina_region_1
            & ~breeder_region_1
            & ~gap_region_1
            & ~furnace_region_1
            & ~he_region_1
            & ~vessel_region_1
            & ~cap_region_1
            & ~heater_region_1
            & ~table_under_source_region_1
            & ~lead_block_1_region_1
            & ~lead_block_2_region_1
            & ~lead_block_3_region_1
            & ~lead_block_4_region_1
            )
    else:
        he_region_1 = (
            +z_plane_5_1
            & -z_plane_12_1
            & -z_cyl_5_1
            & ~epoxy_region_1
            & ~alumina_compressed_region_1
            & ~alumina_region_1
            & ~breeder_region_1
            & ~gap_region_1
            & ~furnace_region_1
            & ~vessel_region_1
            & ~cap_region_1
            & ~heater_region_1
            & ~table_under_source_region_1
            & ~lead_block_1_region_1
            & ~lead_block_2_region_1
            & ~lead_block_3_region_1
            & ~lead_block_4_region_1
            )
        sphere_region_1 = (
            -sphere_1
            & ~epoxy_region_1
            & ~alumina_compressed_region_1
            & ~alumina_region_1
            & ~breeder_region_1
            & ~gap_region_1
            & ~furnace_region_1
            & ~he_region_1
            & ~vessel_region_1
            & ~cap_region_1
            & ~heater_region_1
            & ~table_under_source_region_1
            & ~lead_block_1_region_1
            & ~lead_block_2_region_1
            & ~lead_block_3_region_1
            & ~lead_block_4_region_1
            )

    ########## Cells for BABY 1 ##########
    if source_position == 1:
        source_wall_cell_1 = openmc.Cell(region=source_wall_region)
        source_wall_cell_1.fill = SS304

        source_region = openmc.Cell(region=source_region)
        source_region.fill = None
    
    epoxy_cell_1 = openmc.Cell(region=epoxy_region_1)
    epoxy_cell_1.fill = epoxy

    alumina_compressed_cell_1 = openmc.Cell(region=alumina_compressed_region_1)
    alumina_compressed_cell_1.fill = alumina

    vessel_cell_1 = openmc.Cell(region=vessel_region_1)
    vessel_cell_1.fill = SS316L

    alumina_cell_1 = openmc.Cell(region=alumina_region_1)
    alumina_cell_1.fill = alumina

    breeder_cell_1 = openmc.Cell(region=breeder_region_1)
    if breeder_1 == "Li2O":
        breeder_cell_1.fill = Li2O_bed
    elif breeder_1 == "LiPb":
        breeder_cell_1.fill = lithium_lead
    elif breeder_1 == "ClLiF":
        breeder_cell_1.fill = cllif_nat

    gap_cell_1 = openmc.Cell(region=gap_region_1)
    gap_cell_1.fill = he

    cap_cell_1 = openmc.Cell(region=cap_region_1)
    cap_cell_1.fill = SS316L

    furnace_cell_1 = openmc.Cell(region=furnace_region_1)
    furnace_cell_1.fill = furnace
    
    heater_cell_1 = openmc.Cell(region=heater_region_1)
    heater_cell_1.fill = heater_mat

    table_cell_1 = openmc.Cell(region=table_under_source_region_1)
    table_cell_1.fill = epoxy

    sphere_cell_1 = openmc.Cell(region=sphere_region_1)
    sphere_cell_1.fill = air

    he_cell_1 = openmc.Cell(region=he_region_1)
    he_cell_1.fill = he

    lead_block_1_cell_1 = openmc.Cell(region=lead_block_1_region_1)
    lead_block_1_cell_1.fill = lead

    lead_block_2_cell_1 = openmc.Cell(region=lead_block_2_region_1)
    lead_block_2_cell_1.fill = lead

    lead_block_3_cell_1 = openmc.Cell(region=lead_block_3_region_1)
    lead_block_3_cell_1.fill = lead

    lead_block_4_cell_1 = openmc.Cell(region=lead_block_4_region_1)
    lead_block_4_cell_1.fill = lead

    cells = [
        epoxy_cell_1,
        alumina_compressed_cell_1,
        vessel_cell_1,
        alumina_cell_1,
        cap_cell_1,
        breeder_cell_1,
        gap_cell_1,
        furnace_cell_1,
        heater_cell_1,
        he_cell_1,
        sphere_cell_1,
        table_cell_1,
        lead_block_1_cell_1,
        lead_block_2_cell_1,
        lead_block_3_cell_1,
        lead_block_4_cell_1,
    ]

    if source_position == 1:
        cells.append(source_wall_cell_1)
        cells.append(source_region)     

    breeder_cells = [breeder_cell_1]



    ########## BABY 2 ##########
    x_c, y_c, z_c = baby_positions[1]

    breeder_2 = breeders[1]

    ########## Surfaces ##########
    z_plane_1_2 = openmc.ZPlane(0.0 + z_c)
    z_plane_2_2 = openmc.ZPlane(epoxy_thickness + z_c)
    z_plane_3_2 = openmc.ZPlane(epoxy_thickness + alumina_compressed_thickness + z_c)
    z_plane_4_2 = openmc.ZPlane(
        epoxy_thickness + alumina_compressed_thickness + ov_base_thickness + z_c
    )
    z_plane_5_2 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + alumina_thickness
        + z_c
    )
    z_plane_6_2 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + alumina_thickness
        + he_thickness
        + z_c
    )
    z_plane_7_2 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + alumina_thickness
        + he_thickness
        + iv_base_thickness
        + z_c
    )
    z_plane_8_2 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + alumina_thickness
        + he_thickness
        + iv_base_thickness
        + breeder_thickness
        + z_c
    )
    z_plane_9_2 = openmc.ZPlane(
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
    z_plane_10_2 = openmc.ZPlane(
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
    z_plane_11_2 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + alumina_thickness
        + furnace_thickness
        + z_c
    )
    z_plane_12_2 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + ov_height
        + z_c
    )
    z_plane_13_2 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + ov_height
        + ov_cap
        + z_c
    )
    z_plane_14_2 = openmc.ZPlane(z_c - table_height)
    z_plane_15_2 = openmc.ZPlane(z_c - table_height - epoxy_thickness)

    ########## Cylinders ##########
    z_cyl_1_2 = openmc.ZCylinder(x0=x_c, y0=y_c, r=breeder_radius)
    z_cyl_2_2 = openmc.ZCylinder(x0=x_c, y0=y_c, r=iv_external_radius)
    z_cyl_3_2 = openmc.ZCylinder(x0=x_c, y0=y_c, r=he_radius)
    z_cyl_4_2 = openmc.ZCylinder(x0=x_c, y0=y_c, r=furnace_radius)
    z_cyl_5_2 = openmc.ZCylinder(x0=x_c, y0=y_c, r=ov_internal_radius)
    z_cyl_6_2 = openmc.ZCylinder(x0=x_c, y0=y_c, r=ov_external_radius)

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

    right_cyl_2 = openmc.model.RightCircularCylinder(
        (x_c, y_c, heater_z), heater_length, heater_radius, axis="z"
    )

    if source_position == 2:
        # If BABY 2 is the one with the neutron source, add the source geometry
        source_x = x_c - 13.50
        source_y = y_c
        source_z = z_c - source_z_offset 

        ext_cyl_source = openmc.model.RightCircularCylinder(
        (source_x, source_y, source_z), source_h, source_external_r, axis="x"
        )
        source_region = openmc.model.RightCircularCylinder(
        (source_x + 0.25, source_y, source_z), source_h - 0.50, source_internal_r, axis="x"
        )

        source_wall_region = -ext_cyl_source & +source_region
        source_region = -source_region


    ########## Sphere for trimming geometry ##########
    sphere_2 = openmc.Sphere(x0=x_c, y0=y_c, z0=z_c, r=50.00)  

    ########## Lead bricks positioned under the source ##########
    positions = [
        (x_c - 13.50, y_c, z_c - table_height),
        (x_c - 4.50, y_c, z_c - table_height),
        (x_c + 36.50, y_c, z_c - table_height),
        (x_c + 27.50, y_c, z_c - table_height),
    ]

    lead_blocks_2 = []
    for position in positions:
        lead_block_region_2 = openmc.model.RectangularParallelepiped(
            position[0] - lead_width / 2,
            position[0] + lead_width / 2,
            position[1] - lead_length / 2,
            position[1] + lead_length / 2,
            position[2],
            position[2] + lead_height,
        )
        lead_blocks_2.append(lead_block_region_2)

    ########## Regions for BABY 2 ##########
    epoxy_region_2 = +z_plane_1_2 & -z_plane_2_2 & -sphere_2
    alumina_compressed_region_2 = +z_plane_2_2 & -z_plane_3_2 & -sphere_2
    bottom_vessel_2 = +z_plane_3_2 & -z_plane_4_2 & -z_cyl_6_2
    top_vessel_2 = +z_plane_12_2 & -z_plane_13_2 & -z_cyl_6_2 & +right_cyl_2
    cylinder_vessel_2 = +z_plane_4_2 & -z_plane_12_2 & +z_cyl_5_2 & -z_cyl_6_2
    vessel_region_2 = bottom_vessel_2 | cylinder_vessel_2 | top_vessel_2
    alumina_region_2 = +z_plane_4_2 & -z_plane_5_2 & -z_cyl_5_2
    bottom_cap_2 = +z_plane_6_2 & -z_plane_7_2 & -z_cyl_2_2 & +right_cyl_2
    cylinder_cap_2 = +z_plane_7_2 & -z_plane_9_2 & +z_cyl_1_2 & -z_cyl_2_2 & +right_cyl_2
    top_cap_2 = +z_plane_9_2 & -z_plane_10_2 & -z_cyl_2_2 & +right_cyl_2
    cap_region_2 = bottom_cap_2 | cylinder_cap_2 | top_cap_2

    breeder_region_2 = (
        +z_plane_7_2
        & -z_plane_8_2
        & -z_cyl_1_2
        & +right_cyl_2
    )

    gap_region_2 = +z_plane_8_2 & -z_plane_9_2 & -z_cyl_1_2 & +right_cyl_2
    furnace_region_2 = +z_plane_5_2 & -z_plane_11_2 & +z_cyl_3_2 & -z_cyl_4_2
    heater_region_2 = -right_cyl_2
    table_under_source_region_2 = +z_plane_15_2 & -z_plane_14_2 & -sphere_2
    lead_block_1_region_2 = -lead_blocks_2[0]
    lead_block_2_region_2 = -lead_blocks_2[1]
    lead_block_3_region_2 = -lead_blocks_2[2]
    lead_block_4_region_2 = -lead_blocks_2[3]

    if source_position == 2:
        he_region_2 = (
            +z_plane_5_2
            & -z_plane_12_2
            & -z_cyl_5_2
            & ~source_region
            & ~epoxy_region_2
            & ~alumina_compressed_region_2
            & ~alumina_region_2
            & ~breeder_region_2
            & ~gap_region_2
            & ~furnace_region_2
            & ~vessel_region_2
            & ~cap_region_2
            & ~heater_region_2
            & ~table_under_source_region_2
            & ~lead_block_1_region_2
            & ~lead_block_2_region_2
            & ~lead_block_3_region_2
            & ~lead_block_4_region_2
            )
        sphere_region_2 = (
            -sphere_2
            & ~source_wall_region
            & ~source_region
            & ~epoxy_region_2
            & ~alumina_compressed_region_2
            & ~alumina_region_2
            & ~breeder_region_2
            & ~gap_region_2
            & ~furnace_region_2
            & ~he_region_2
            & ~vessel_region_2
            & ~cap_region_2
            & ~heater_region_2
            & ~table_under_source_region_2
            & ~lead_block_1_region_2
            & ~lead_block_2_region_2
            & ~lead_block_3_region_2
            & ~lead_block_4_region_2
            )
    else:
        he_region_2 = (
            +z_plane_5_2
            & -z_plane_12_2
            & -z_cyl_5_2
            & ~epoxy_region_2
            & ~alumina_compressed_region_2
            & ~alumina_region_2
            & ~breeder_region_2
            & ~gap_region_2
            & ~furnace_region_2
            & ~vessel_region_2
            & ~cap_region_2
            & ~heater_region_2
            & ~table_under_source_region_2
            & ~lead_block_1_region_2
            & ~lead_block_2_region_2
            & ~lead_block_3_region_2
            & ~lead_block_4_region_2
            )
        sphere_region_2 = (
            -sphere_2
            & ~epoxy_region_2
            & ~alumina_compressed_region_2
            & ~alumina_region_2
            & ~breeder_region_2
            & ~gap_region_2
            & ~furnace_region_2
            & ~he_region_2
            & ~vessel_region_2
            & ~cap_region_2
            & ~heater_region_2
            & ~table_under_source_region_2
            & ~lead_block_1_region_2
            & ~lead_block_2_region_2
            & ~lead_block_3_region_2
            & ~lead_block_4_region_2
            )

    ########## Cells for BABY 2 ##########
    if source_position == 2:
        source_wall_cell_1 = openmc.Cell(region=source_wall_region)
        source_wall_cell_1.fill = SS304

        source_region = openmc.Cell(region=source_region)
        source_region.fill = None
    
    epoxy_cell_2 = openmc.Cell(region=epoxy_region_2)
    epoxy_cell_2.fill = epoxy

    alumina_compressed_cell_2 = openmc.Cell(region=alumina_compressed_region_2)
    alumina_compressed_cell_2.fill = alumina

    vessel_cell_2 = openmc.Cell(region=vessel_region_2)
    vessel_cell_2.fill = SS316L

    alumina_cell_2 = openmc.Cell(region=alumina_region_2)
    alumina_cell_2.fill = alumina

    breeder_cell_2 = openmc.Cell(region=breeder_region_2)
    if breeder_2 == "Li2O":
        breeder_cell_2.fill = Li2O_bed
    elif breeder_2 == "LiPb":
        breeder_cell_2.fill = lithium_lead
    elif breeder_2 == "ClLiF":
        breeder_cell_2.fill = cllif_nat

    gap_cell_2 = openmc.Cell(region=gap_region_2)
    gap_cell_2.fill = he

    cap_cell_2 = openmc.Cell(region=cap_region_2)
    cap_cell_2.fill = SS316L

    furnace_cell_2 = openmc.Cell(region=furnace_region_2)
    furnace_cell_2.fill = furnace
    
    heater_cell_2 = openmc.Cell(region=heater_region_2)
    heater_cell_2.fill = heater_mat

    table_cell_2 = openmc.Cell(region=table_under_source_region_2)
    table_cell_2.fill = epoxy

    sphere_cell_2 = openmc.Cell(region=sphere_region_2)
    sphere_cell_2.fill = air

    he_cell_2 = openmc.Cell(region=he_region_2)
    he_cell_2.fill = he

    lead_block_1_cell_2 = openmc.Cell(region=lead_block_1_region_2)
    lead_block_1_cell_2.fill = lead

    lead_block_2_cell_2 = openmc.Cell(region=lead_block_2_region_2)
    lead_block_2_cell_2.fill = lead

    lead_block_3_cell_2 = openmc.Cell(region=lead_block_3_region_2)
    lead_block_3_cell_2.fill = lead

    lead_block_4_cell_2 = openmc.Cell(region=lead_block_4_region_2)
    lead_block_4_cell_2.fill = lead

    cells.append(epoxy_cell_2)
    cells.append(alumina_compressed_cell_2)
    cells.append(vessel_cell_2)
    cells.append(alumina_cell_2)
    cells.append(cap_cell_2)
    cells.append(breeder_cell_2)
    cells.append(gap_cell_2)
    cells.append(furnace_cell_2)
    cells.append(heater_cell_2)
    cells.append(he_cell_2)
    cells.append(sphere_cell_2)
    cells.append(table_cell_2)
    cells.append(lead_block_1_cell_2)
    cells.append(lead_block_2_cell_2)
    cells.append(lead_block_3_cell_2)
    cells.append(lead_block_4_cell_2)

    if source_position == 2:
        cells.append(source_wall_cell_1)
        cells.append(source_region)     

    breeder_cells.append(breeder_cell_2)



    ########## BABY 3 ##########
    x_c, y_c, z_c = baby_positions[2]

    breeder_3 = breeders[2]

    ########## Surfaces ##########
    z_plane_1_3 = openmc.ZPlane(0.0 + z_c)
    z_plane_2_3 = openmc.ZPlane(epoxy_thickness + z_c)
    z_plane_3_3 = openmc.ZPlane(epoxy_thickness + alumina_compressed_thickness + z_c)
    z_plane_4_3 = openmc.ZPlane(
        epoxy_thickness + alumina_compressed_thickness + ov_base_thickness + z_c
    )
    z_plane_5_3 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + alumina_thickness
        + z_c
    )
    z_plane_6_3 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + alumina_thickness
        + he_thickness
        + z_c
    )
    z_plane_7_3 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + alumina_thickness
        + he_thickness
        + iv_base_thickness
        + z_c
    )
    z_plane_8_3 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + alumina_thickness
        + he_thickness
        + iv_base_thickness
        + breeder_thickness
        + z_c
    )
    z_plane_9_3 = openmc.ZPlane(
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
    z_plane_10_3 = openmc.ZPlane(
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
    z_plane_11_3 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + alumina_thickness
        + furnace_thickness
        + z_c
    )
    z_plane_12_3 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + ov_height
        + z_c
    )
    z_plane_13_3 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + ov_height
        + ov_cap
        + z_c
    )
    z_plane_14_3 = openmc.ZPlane(z_c - table_height)
    z_plane_15_3 = openmc.ZPlane(z_c - table_height - epoxy_thickness)

    ########## Cylinders ##########
    z_cyl_1_3 = openmc.ZCylinder(x0=x_c, y0=y_c, r=breeder_radius)
    z_cyl_2_3 = openmc.ZCylinder(x0=x_c, y0=y_c, r=iv_external_radius)
    z_cyl_3_3 = openmc.ZCylinder(x0=x_c, y0=y_c, r=he_radius)
    z_cyl_4_3 = openmc.ZCylinder(x0=x_c, y0=y_c, r=furnace_radius)
    z_cyl_5_3 = openmc.ZCylinder(x0=x_c, y0=y_c, r=ov_internal_radius)
    z_cyl_6_3 = openmc.ZCylinder(x0=x_c, y0=y_c, r=ov_external_radius)

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

    right_cyl_3 = openmc.model.RightCircularCylinder(
        (x_c, y_c, heater_z), heater_length, heater_radius, axis="z"
    )

    if source_position == 3:
        # If BABY 3 is the one with the neutron source, add the source geometry
        source_x = x_c - 13.50
        source_y = y_c
        source_z = z_c - source_z_offset 

        ext_cyl_source = openmc.model.RightCircularCylinder(
        (source_x, source_y, source_z), source_h, source_external_r, axis="x"
        )
        source_region = openmc.model.RightCircularCylinder(
        (source_x + 0.25, source_y, source_z), source_h - 0.50, source_internal_r, axis="x"
        )

        source_wall_region = -ext_cyl_source & +source_region
        source_region = -source_region


    ########## Sphere for trimming geometry ##########
    sphere_3 = openmc.Sphere(x0=x_c, y0=y_c, z0=z_c, r=50.00)  

    ########## Lead bricks positioned under the source ##########
    positions = [
        (x_c - 13.50, y_c, z_c - table_height),
        (x_c - 4.50, y_c, z_c - table_height),
        (x_c + 36.50, y_c, z_c - table_height),
        (x_c + 27.50, y_c, z_c - table_height),
    ]

    lead_blocks_3 = []
    for position in positions:
        lead_block_region_3 = openmc.model.RectangularParallelepiped(
            position[0] - lead_width / 2,
            position[0] + lead_width / 2,
            position[1] - lead_length / 2,
            position[1] + lead_length / 2,
            position[2],
            position[2] + lead_height,
        )
        lead_blocks_3.append(lead_block_region_3)

    ########## Regions for BABY 2 ##########
    epoxy_region_3 = +z_plane_1_3 & -z_plane_2_3 & -sphere_3
    alumina_compressed_region_3 = +z_plane_2_3 & -z_plane_3_3 & -sphere_3
    bottom_vessel_3 = +z_plane_3_3 & -z_plane_4_3 & -z_cyl_6_3
    top_vessel_3 = +z_plane_12_3 & -z_plane_13_3 & -z_cyl_6_3 & +right_cyl_3
    cylinder_vessel_3 = +z_plane_4_3 & -z_plane_12_3 & +z_cyl_5_3 & -z_cyl_6_3
    vessel_region_3 = bottom_vessel_3 | cylinder_vessel_3 | top_vessel_3
    alumina_region_3 = +z_plane_4_3 & -z_plane_5_3 & -z_cyl_5_3
    bottom_cap_3 = +z_plane_6_3 & -z_plane_7_3 & -z_cyl_2_3 & +right_cyl_3
    cylinder_cap_3 = +z_plane_7_3 & -z_plane_9_3 & +z_cyl_1_3 & -z_cyl_2_3 & +right_cyl_3
    top_cap_3 = +z_plane_9_3 & -z_plane_10_3 & -z_cyl_2_3 & +right_cyl_3
    cap_region_3 = bottom_cap_3 | cylinder_cap_3 | top_cap_3

    breeder_region_3 = (
        +z_plane_7_3
        & -z_plane_8_3
        & -z_cyl_1_3
        & +right_cyl_3
    )

    gap_region_3 = +z_plane_8_3 & -z_plane_9_3 & -z_cyl_1_3 & +right_cyl_3
    furnace_region_3 = +z_plane_5_3 & -z_plane_11_3 & +z_cyl_3_3 & -z_cyl_4_3
    heater_region_3 = -right_cyl_3
    table_under_source_region_3 = +z_plane_15_3 & -z_plane_14_3 & -sphere_3
    lead_block_1_region_3 = -lead_blocks_3[0]
    lead_block_2_region_3 = -lead_blocks_3[1]
    lead_block_3_region_3 = -lead_blocks_3[2]
    lead_block_4_region_3 = -lead_blocks_3[3]

    if source_position == 3:
        he_region_3 = (
            +z_plane_5_3
            & -z_plane_12_3
            & -z_cyl_5_3
            & ~source_region
            & ~epoxy_region_3
            & ~alumina_compressed_region_3
            & ~alumina_region_3
            & ~breeder_region_3
            & ~gap_region_3
            & ~furnace_region_3
            & ~vessel_region_3
            & ~cap_region_3
            & ~heater_region_3
            & ~table_under_source_region_3
            & ~lead_block_1_region_3
            & ~lead_block_2_region_3
            & ~lead_block_3_region_3
            & ~lead_block_4_region_3
            )
        sphere_region_3 = (
            -sphere_3
            & ~source_wall_region
            & ~source_region
            & ~epoxy_region_3
            & ~alumina_compressed_region_3
            & ~alumina_region_3
            & ~breeder_region_3
            & ~gap_region_3
            & ~furnace_region_3
            & ~he_region_3
            & ~vessel_region_3
            & ~cap_region_3
            & ~heater_region_3
            & ~table_under_source_region_3
            & ~lead_block_1_region_3
            & ~lead_block_2_region_3
            & ~lead_block_3_region_3
            & ~lead_block_4_region_3
            )
    else:
        he_region_3 = (
            +z_plane_5_3
            & -z_plane_12_3
            & -z_cyl_5_3
            & ~epoxy_region_3
            & ~alumina_compressed_region_3
            & ~alumina_region_3
            & ~breeder_region_3
            & ~gap_region_3
            & ~furnace_region_3
            & ~vessel_region_3
            & ~cap_region_3
            & ~heater_region_3
            & ~table_under_source_region_3
            & ~lead_block_1_region_3
            & ~lead_block_2_region_3
            & ~lead_block_3_region_3
            & ~lead_block_4_region_3
            )
        sphere_region_3 = (
            -sphere_3
            & ~epoxy_region_3
            & ~alumina_compressed_region_3
            & ~alumina_region_3
            & ~breeder_region_3
            & ~gap_region_3
            & ~furnace_region_3
            & ~he_region_3
            & ~vessel_region_3
            & ~cap_region_3
            & ~heater_region_3
            & ~table_under_source_region_3
            & ~lead_block_1_region_3
            & ~lead_block_2_region_3
            & ~lead_block_3_region_3
            & ~lead_block_4_region_3
            )

    ########## Cells for BABY 3 ##########
    if source_position == 3:
        source_wall_cell_1 = openmc.Cell(region=source_wall_region)
        source_wall_cell_1.fill = SS304

        source_region = openmc.Cell(region=source_region)
        source_region.fill = None
    
    epoxy_cell_3 = openmc.Cell(region=epoxy_region_3)
    epoxy_cell_3.fill = epoxy

    alumina_compressed_cell_3 = openmc.Cell(region=alumina_compressed_region_3)
    alumina_compressed_cell_3.fill = alumina

    vessel_cell_3 = openmc.Cell(region=vessel_region_3)
    vessel_cell_3.fill = SS316L

    alumina_cell_3 = openmc.Cell(region=alumina_region_3)
    alumina_cell_3.fill = alumina

    breeder_cell_3 = openmc.Cell(region=breeder_region_3)
    if breeder_3 == "Li2O":
        breeder_cell_3.fill = Li2O_bed
    elif breeder_3 == "LiPb":
        breeder_cell_3.fill = lithium_lead
    elif breeder_3 == "ClLiF":
        breeder_cell_3.fill = cllif_nat

    gap_cell_3 = openmc.Cell(region=gap_region_3)
    gap_cell_3.fill = he

    cap_cell_3 = openmc.Cell(region=cap_region_3)
    cap_cell_3.fill = SS316L

    furnace_cell_3 = openmc.Cell(region=furnace_region_3)
    furnace_cell_3.fill = furnace
    
    heater_cell_3 = openmc.Cell(region=heater_region_3)
    heater_cell_3.fill = heater_mat

    table_cell_3 = openmc.Cell(region=table_under_source_region_3)
    table_cell_3.fill = epoxy

    sphere_cell_3 = openmc.Cell(region=sphere_region_3)
    sphere_cell_3.fill = air

    he_cell_3 = openmc.Cell(region=he_region_3)
    he_cell_3.fill = he

    lead_block_1_cell_3 = openmc.Cell(region=lead_block_1_region_3)
    lead_block_1_cell_3.fill = lead

    lead_block_2_cell_3 = openmc.Cell(region=lead_block_2_region_3)
    lead_block_2_cell_3.fill = lead

    lead_block_3_cell_3 = openmc.Cell(region=lead_block_3_region_3)
    lead_block_3_cell_3.fill = lead

    lead_block_4_cell_3 = openmc.Cell(region=lead_block_4_region_3)
    lead_block_4_cell_3.fill = lead

    cells.append(epoxy_cell_3)
    cells.append(alumina_compressed_cell_3)
    cells.append(vessel_cell_3)
    cells.append(alumina_cell_3)
    cells.append(cap_cell_3)
    cells.append(breeder_cell_3)
    cells.append(gap_cell_3)
    cells.append(furnace_cell_3)
    cells.append(heater_cell_3)
    cells.append(he_cell_3)
    cells.append(sphere_cell_3)
    cells.append(table_cell_3)
    cells.append(lead_block_1_cell_3)
    cells.append(lead_block_2_cell_3)
    cells.append(lead_block_3_cell_3)
    cells.append(lead_block_4_cell_3)

    if source_position == 3:
        cells.append(source_wall_cell_1)
        cells.append(source_region)     

    breeder_cells.append(breeder_cell_3)

    ########## Global sphere to enclose all BABY geometries ##########
    global_sphere = sphere_geometry(baby_positions)

    outer_region = -global_sphere & +sphere_1 & +sphere_2 & +sphere_3
    outer_cell = openmc.Cell(region=outer_region)
    outer_cell.fill = air

    cells.append(outer_cell)

    return cells, breeder_cells


############################################################################
# Dimensions
# All dimensions in cm

## List of BABY coordinates within vault
baby_positions = [
    (587, 60, 100),  # BABY 1 center
    (885, 60, 100),  # BABY 2 center
    (897, 291, 100),  # BABY 2 center
]

# Breeder materials for each BABY experiment
# The order of the breeders should match the order of the BABY positions
breeders = [
    "Li2O",
    "Li2O",
    "Li2O"
]

## Source position
source_position = 1  # Index of the BABY position where the source is located
source_z_offset = 5.635  # Offset for the source Z position

source_x = baby_positions[source_position-1][0] - 13.50
source_y = baby_positions[source_position-1][1]
source_z = baby_positions[source_position-1][2] - source_z_offset 

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

print("**Li2O packed bed properties**")
print("Oxygen mass fraction:", O_mass_frac_bed, "% mass")
print("Lithium mass fraction:", Li_mass_frac_bed, "% mass")
print("Helium mass fraction:", He_mass_frac_bed, "% mass")
print("Pellet bed density:", pellet_bed_density, "g/cm3")

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

if __name__ == "__main__":

    model = nursery_model()
    model.run()
    sp = openmc.StatePoint(f"statepoint.{model.settings.batches}.h5")

    print("Nursery model simulation completed successfully.")
    print("BABY locations and breeder materials:")
    for i, (pos, breeder) in enumerate(zip(baby_positions, breeders), start=1):
        print(f"  BABY {i} at position {pos} cm with breeder material: {breeder}")


    ## BABY 1 results
    tbr_tally_1 = sp.get_tally(name="TBR_1").get_pandas_dataframe()

    mean_1 = tbr_tally_1["mean"].iloc[0]
    stdev_1 = tbr_tally_1["std. dev."].iloc[0]

    print(f"BABY 1 TBR: {mean_1:.6e}\n")
    print(f"BABY 1 TBR std. dev: {stdev_1:.6e}\n")

    rel_stdev_1 = stdev_1 / mean_1

    print(f"BABY 1 Relative standard deviation: {rel_stdev_1:.6e}\n")

    ## BABY 2 results
    tbr_tally_2 = sp.get_tally(name="TBR_2").get_pandas_dataframe()

    mean_2 = tbr_tally_2["mean"].iloc[0]
    stdev_2 = tbr_tally_2["std. dev."].iloc[0]

    print(f"BABY 2 TBR: {mean_2:.6e}\n")
    print(f"BABY 2 TBR std. dev: {stdev_2:.6e}\n")

    rel_stdev_2 = stdev_2 / mean_2

    print(f"BABY 2 Relative standard deviation: {rel_stdev_2:.6e}\n")

    ## BABY 3 results
    tbr_tally_3 = sp.get_tally(name="TBR_3").get_pandas_dataframe()

    mean_3 = tbr_tally_3["mean"].iloc[0]
    stdev_3 = tbr_tally_3["std. dev."].iloc[0]

    print(f"BABY 3 TBR: {mean_3:.6e}\n")
    print(f"BABY 3 TBR std. dev: {stdev_3:.6e}\n")

    rel_stdev_3 = stdev_3 / mean_3

    print(f"BABY 3 Relative standard deviation: {rel_stdev_3:.6e}\n")

    print("Relative standard deviation below 1e-02 (1%) indicates good convergence.")

    processed_data = {
        "modelled_TBR_1": {
            "mean": tbr_tally_1["mean"].iloc[0],
            "std_dev": tbr_tally_1["std. dev."].iloc[0],
        },
        "modelled_TBR_2": {
        "mean": tbr_tally_2["mean"].iloc[0],
        "std_dev": tbr_tally_2["std. dev."].iloc[0],
        },
        "modelled_TBR_3": {
        "mean": tbr_tally_3["mean"].iloc[0],
        "std_dev": tbr_tally_3["std. dev."].iloc[0],
        }
    }

    import json

    processed_data_file = "../../data/processed_data.json"

    try:
        with open(processed_data_file, "r") as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        print(f"Processed data file not found, creating it in {processed_data_file}")
        existing_data = {}

    existing_data.update(processed_data)

    with open(processed_data_file, "w") as f:
        json.dump(existing_data, f, indent=4)

    print(f"Processed data stored in {processed_data_file}")

