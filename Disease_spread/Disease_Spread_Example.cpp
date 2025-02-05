//!#####################################################################
//! \file Disease_Spread_Example.cpp
//!#####################################################################
#include <nova/Dynamics/Hierarchy/Grid_Hierarchy_Initializer.h>
#include <nova/Dynamics/Hierarchy/Grid_Hierarchy_Iterator.h>
#include <nova/Dynamics/Hierarchy/Advection/Grid_Hierarchy_Advection.h>
#include <nova/Dynamics/Hierarchy/Interpolation/Grid_Hierarchy_Averaging.h>
#include <nova/Tools/Grids/Grid_Iterator_Face.h>
#include <nova/Tools/Krylov_Solvers/Conjugate_Gradient.h>
#include <nova/Tools/Utilities/File_Utilities.h>
//#include "Apply_Pressure.h"
//#include "Boundary_Value_Initializer.h"
#include "CG_code/CG_System.h"
//#include "Compute_Time_Step.h"
//#include "Density_Modifier.h"
//#include "Initialize_Dirichlet_Cells.h"
//#include "Multigrid_Data.h"
#include "Disease_Spread_Example.h"
#include <omp.h>
using namespace Nova;
namespace Nova{
extern int number_of_threads;
}
//######################################################################
// Constructor
//######################################################################
template<class T,int d> Disease_Spread_Example<T,d>::
Disease_Spread_Example()
    :Base(),hierarchy(nullptr),rasterizer(nullptr)
{
    //TODO these are variable that appear in poisson_data.h. I will need to specify my own version.
    face_velocity_channels(0)           = &Struct_type::ch0;
    face_velocity_channels(1)           = &Struct_type::ch1;
    if(d==3) face_velocity_channels(2)  = &Struct_type::ch2;
    pressure_channel                    = &Struct_type::ch3;
    density_channel                     = &Struct_type::ch4;
    
    /*
    What will the variables be?
    To start, I have two separate things: air following Navier-Stokes, and disease concentration following Convection-Diffusion. The former acts on the latter.
    I will need face velocity channels and pressure channels for air, and density_channel will have to be renamed disease_concentration_chnnel.
    I will also need to reconfigure the code.
    */
}
//######################################################################
// Initialize
//######################################################################
template<class T,int d> void Disease_Spread_Example<T,d>::
Initialize()
{
    Initialize_SPGrid();
    Initialize_Fluid_State();
}
//######################################################################
// Initialize_SPGrid
//######################################################################
template<class T,int d> void Disease_Spread_Example<T,d>::
Initialize_SPGrid()
{
    /*Log::Scope scope("Initialize_SPGrid");
    Initialize_Rasterizer();
    for(Grid_Hierarchy_Iterator<d,Hierarchy_Rasterizer> iterator(hierarchy->Lattice(levels-1).Cell_Indices(),levels-1,*rasterizer);iterator.Valid();iterator.Next());
    Grid_Hierarchy_Initializer<Struct_type,T,d>::Flag_Ghost_Cells(*hierarchy);
    Grid_Hierarchy_Initializer<Struct_type,T,d>::Flag_Valid_Faces(*hierarchy);
    Grid_Hierarchy_Initializer<Struct_type,T,d>::Flag_Active_Faces(*hierarchy);
    Grid_Hierarchy_Initializer<Struct_type,T,d>::Flag_Active_Nodes(*hierarchy);
    Grid_Hierarchy_Initializer<Struct_type,T,d>::Flag_Shared_Nodes(*hierarchy);
    Grid_Hierarchy_Initializer<Struct_type,T,d>::Flag_Ghost_Nodes(*hierarchy);
    Grid_Hierarchy_Initializer<Struct_type,T,d>::Flag_T_Junction_Nodes(*hierarchy);
    Initialize_Dirichlet_Cells<Struct_type,T,d>(*hierarchy,domain_walls);
    //Set_Neumann_Faces_Inside_Sources();
    hierarchy->Update_Block_Offsets();
    hierarchy->Initialize_Red_Black_Partition(2*number_of_threads);*/
}
//######################################################################
// Limit_Dt
//######################################################################
template<class T,int d> void Disease_Spread_Example<T,d>::
Limit_Dt(T& dt,const T time)
{
    /*T dt_convection=(T)0.;

    Vector<uint64_t,d> other_face_offsets;
    for(int axis=0;axis<d;++axis) other_face_offsets(axis)=Topology_Helper::Axis_Vector_Offset(axis);

    for(int level=0;level<levels;++level)
        Compute_Time_Step<Struct_type,T,d>(*hierarchy,hierarchy->Blocks(level),face_velocity_channels,
                                           other_face_offsets,level,dt_convection);

    if(dt_convection>(T)1e-5) dt=cfl/dt_convection;
    Log::cout<<"Time Step: "<<dt<<std::endl;*/
}
//######################################################################
// Advect_Density
//######################################################################
template<class T,int d> void Disease_Spread_Example<T,d>::
Advect_Density(const T dt)
{
    Channel_Vector node_velocity_channels;
    node_velocity_channels(0)               = &Struct_type::ch5;
    node_velocity_channels(1)               = &Struct_type::ch6;
    if(d==3) node_velocity_channels(2)      = &Struct_type::ch7;
    T Struct_type::* node_density_channel   = &Struct_type::ch8;
    T Struct_type::* temp_channel           = &Struct_type::ch9;
    Grid_Hierarchy_Averaging<Struct_type,T,d>::Average_Cell_Density_To_Nodes(*hierarchy,density_channel,node_density_channel,temp_channel);
    Grid_Hierarchy_Averaging<Struct_type,T,d>::Average_Face_Velocities_To_Nodes(*hierarchy,face_velocity_channels,node_velocity_channels,temp_channel);
    Grid_Hierarchy_Advection<Struct_type,T,d>::Advect_Density(*hierarchy,node_velocity_channels,density_channel,node_density_channel,temp_channel,dt);
}
//######################################################################
// Modify_Density_With_Sources
//######################################################################
template<class T,int d> void Disease_Spread_Example<T,d>::
Modify_Density_With_Sources()
{
    /*for(int level=0;level<levels;++level)
        Density_Modifier<Struct_type,T,d>(*hierarchy,hierarchy->Blocks(level),density_channel,sources,level);*/
}
//######################################################################
// Advect_Face_Velocities
//######################################################################
template<class T,int d> void Disease_Spread_Example<T,d>::
Advect_Face_Velocities(const T dt)
{
    Channel_Vector node_velocity_channels;
    node_velocity_channels(0)               = &Struct_type::ch5;
    node_velocity_channels(1)               = &Struct_type::ch6;
    if(d==3) node_velocity_channels(2)      = &Struct_type::ch7;
    T Struct_type::* temp_channel           = &Struct_type::ch9;
    //Grid_Hierarchy_Averaging<Struct_type,T,d>::Average_Face_Velocities_To_Nodes(*hierarchy,face_velocity_channels,node_velocity_channels,temp_channel);
    Grid_Hierarchy_Advection<Struct_type,T,d>::Advect_Face_Velocities(*hierarchy,face_velocity_channels,node_velocity_channels,temp_channel,dt);
}
//######################################################################
// Set_Neumann_Faces_Inside_Sources
//######################################################################
template<class T,int d> void Disease_Spread_Example<T,d>::
Set_Neumann_Faces_Inside_Sources()
{
    for(int level=0;level<levels;++level){
        auto flags=hierarchy->Allocator(level).template Get_Array<Struct_type,unsigned>(&Struct_type::flags);
        for(Grid_Iterator_Face<T,d> iterator(hierarchy->Lattice(level));iterator.Valid();iterator.Next()){
            const int axis=iterator.Axis();const T_INDEX& face_index=iterator.Face_Index();
            uint64_t face_offset=Flag_array_mask::Linear_Offset(face_index._data);
            const uint64_t face_active_mask=Topology_Helper::Face_Active_Mask(axis);
            const uint64_t face_valid_mask=Topology_Helper::Face_Valid_Mask(axis);
            const TV X=hierarchy->Lattice(level).Face(axis,face_index);
            for(size_t i=0;i<sources.size();++i) if(sources(i)->Inside(X)){
                if(hierarchy->template Set<unsigned>(level,&Struct_type::flags).Is_Set(face_offset,face_valid_mask))
                    flags(face_offset)&=~face_active_mask;
                break;}}}
}
//######################################################################
// Initialize_Values_At_Boundary_Conditions
//######################################################################
template<class T,int d> void Disease_Spread_Example<T,d>::
Initialize_Values_At_Boundary_Conditions()
{
    /*for(int level=0;level<levels;++level)
        Boundary_Value_Initializer<Struct_type,T,d>(*hierarchy,hierarchy->Blocks(level),sources,
                                                    face_velocity_channels,pressure_channel,level);*/
}
//######################################################################
// Set_Boundary_Conditions
//######################################################################
template<class T,int d> void Disease_Spread_Example<T,d>::
Set_Boundary_Conditions()
{
    Initialize_Values_At_Boundary_Conditions();
}
//######################################################################
// Project
//######################################################################
template<class T,int d> void Disease_Spread_Example<T,d>::
Project(const T dt)
{
    /*using Multigrid_struct_type             = Multigrid_Data<T>;
    using Hierarchy_Projection              = Grid_Hierarchy_Projection<Struct_type,T,d>;

    // set up divergence channel
    T Struct_type::* divergence_channel     = &Struct_type::ch5;

    // clear
    for(int level=0;level<levels;++level){
        SPGrid::Clear<Struct_type,T,d>(hierarchy->Allocator(level),hierarchy->Blocks(level),pressure_channel);
        SPGrid::Clear<Struct_type,T,d>(hierarchy->Allocator(level),hierarchy->Blocks(level),divergence_channel);}

    // set boundary conditions
    Set_Boundary_Conditions();

    // compute divergence
    Hierarchy_Projection::Compute_Divergence(*hierarchy,face_velocity_channels,divergence_channel);

    Channel_Vector gradient_channels;
    gradient_channels(0)                    = &Struct_type::ch6;
    gradient_channels(1)                    = &Struct_type::ch7;
    if(d==3) gradient_channels(2)           = &Struct_type::ch8;

    CG_System<Struct_type,Multigrid_struct_type,T,d> cg_system(*hierarchy,gradient_channels,mg_levels,3,1,200);

    T Struct_type::* q_channel              = &Struct_type::ch9;
    T Struct_type::* r_channel              = &Struct_type::ch10;
    T Struct_type::* s_channel              = &Struct_type::ch11;
    T Struct_type::* k_channel              = &Struct_type::ch11;
    T Struct_type::* z_channel              = &Struct_type::ch12;

    // clear all channels
    for(int level=0;level<levels;++level){
        SPGrid::Clear<Struct_type,T,d>(hierarchy->Allocator(level),hierarchy->Blocks(level),q_channel);
        SPGrid::Clear<Struct_type,T,d>(hierarchy->Allocator(level),hierarchy->Blocks(level),r_channel);
        SPGrid::Clear<Struct_type,T,d>(hierarchy->Allocator(level),hierarchy->Blocks(level),s_channel);
        SPGrid::Clear<Struct_type,T,d>(hierarchy->Allocator(level),hierarchy->Blocks(level),z_channel);}

    CG_Vector<Struct_type,T,d> x_V(*hierarchy,pressure_channel);
    CG_Vector<Struct_type,T,d> b_V(*hierarchy,divergence_channel);
    CG_Vector<Struct_type,T,d> q_V(*hierarchy,q_channel);
    CG_Vector<Struct_type,T,d> r_V(*hierarchy,r_channel);
    CG_Vector<Struct_type,T,d> s_V(*hierarchy,s_channel);
    CG_Vector<Struct_type,T,d> k_V(*hierarchy,k_channel);
    CG_Vector<Struct_type,T,d> z_V(*hierarchy,z_channel);

    Conjugate_Gradient<T> cg;
    cg_system.Multiply(x_V,r_V);
    r_V-=b_V;
    const T b_norm=cg_system.Convergence_Norm(r_V);
    Log::cout<<"Norm: "<<b_norm<<std::endl;
    cg.print_residuals=true;
    cg.print_diagnostics=true;
    cg.restart_iterations=cg_restart_iterations;
    const T tolerance=std::max(cg_tolerance*b_norm,(T)1e-6);
    cg.Solve(cg_system,x_V,b_V,q_V,s_V,r_V,k_V,z_V,tolerance,0,cg_iterations);

    // compute gradients
    Hierarchy_Projection::Compute_Gradient(*hierarchy,gradient_channels,pressure_channel);

    for(int level=0;level<levels;++level)
        Apply_Pressure<Struct_type,T,d>(*hierarchy,hierarchy->Blocks(level),face_velocity_channels,
                                        gradient_channels,pressure_channel,level);*/
}
//######################################################################
// Register_Options
//######################################################################
template<class T,int d> void Disease_Spread_Example<T,d>::
Register_Options()
{
    Base::Register_Options();

    parse_args->Add_Double_Argument("-cfl",(T).5,"CFL number.");
    parse_args->Add_Integer_Argument("-levels",1,"Number of levels in the SPGrid hierarchy.");
    parse_args->Add_Integer_Argument("-mg_levels",1,"Number of levels in the Multigrid hierarchy.");
    parse_args->Add_Integer_Argument("-threads",1,"Number of threads for OpenMP to use");
    if(d==2) parse_args->Add_Vector_2D_Argument("-size",Vector<double,2>(64.),"n","Grid resolution");
    else if(d==3) parse_args->Add_Vector_3D_Argument("-size",Vector<double,3>(64.),"n","Grid resolution");

    // for CG
    parse_args->Add_Integer_Argument("-cg_iterations",100,"Number of CG iterations.");
    parse_args->Add_Integer_Argument("-cg_restart_iterations",40,"Number of CG restart iterations.");
    parse_args->Add_Double_Argument("-cg_tolerance",1e-4,"CG tolerance");
}
//######################################################################
// Parse_Options
//######################################################################
template<class T,int d> void Disease_Spread_Example<T,d>::
Parse_Options()
{
    Base::Parse_Options();

    cfl=(T)parse_args->Get_Double_Value("-cfl");
    levels=parse_args->Get_Integer_Value("-levels");
    mg_levels=parse_args->Get_Integer_Value("-mg_levels");
    number_of_threads=parse_args->Get_Integer_Value("-threads");
    omp_set_num_threads(number_of_threads);
    if(d==2){auto cell_counts_2d=parse_args->Get_Vector_2D_Value("-size");for(int v=0;v<d;++v) counts(v)=cell_counts_2d(v);}
    else{auto cell_counts_3d=parse_args->Get_Vector_3D_Value("-size");for(int v=0;v<d;++v) counts(v)=cell_counts_3d(v);}

    cg_iterations=parse_args->Get_Integer_Value("-cg_iterations");
    cg_restart_iterations=parse_args->Get_Integer_Value("-cg_restart_iterations");
    cg_tolerance=(T)parse_args->Get_Double_Value("-cg_tolerance");
}
//######################################################################
// Write_Output_Files
//######################################################################
template<class T,int d> void Disease_Spread_Example<T,d>::
Write_Output_Files(const int frame) const
{
    File_Utilities::Create_Directory(output_directory+"/"+std::to_string(frame));
    File_Utilities::Write_To_Text_File(output_directory+"/"+std::to_string(frame)+"/frame_title",frame_title);

    // write hierarchy
    File_Utilities::Write_To_Text_File(output_directory+"/"+std::to_string(frame)+"/levels",levels);
    hierarchy->Write_Hierarchy(output_directory,frame);
    hierarchy->template Write_Channel<T>(output_directory+"/"+std::to_string(frame)+"/spgrid_density",density_channel);
    hierarchy->template Write_Channel<T>(output_directory+"/"+std::to_string(frame)+"/spgrid_u",face_velocity_channels(0));
    hierarchy->template Write_Channel<T>(output_directory+"/"+std::to_string(frame)+"/spgrid_v",face_velocity_channels(1));
    if(d==3) hierarchy->template Write_Channel<T>(output_directory+"/"+std::to_string(frame)+"/spgrid_w",face_velocity_channels(2));
}
//######################################################################
// Read_Output_Files
//######################################################################
template<class T,int d> void Disease_Spread_Example<T,d>::
Read_Output_Files(const int frame)
{
}
//######################################################################
template class Nova::Disease_Spread_Example<float,2>;
template class Nova::Disease_Spread_Example<float,3>;
#ifdef COMPILE_WITH_DOUBLE_SUPPORT
template class Nova::Disease_Spread_Example<double,2>;
template class Nova::Disease_Spread_Example<double,3>;
#endif
