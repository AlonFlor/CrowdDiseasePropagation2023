//!#####################################################################
//! \file CG_System.h
//!#####################################################################
// Class CG_System
//######################################################################
#ifndef __CG_System__
#define __CG_System__

#include <nova/Tools/Krylov_Solvers/Krylov_System_Base.h>
#include "CG_Vector.h"
#include "Convergence_Norm_Helper.h"
#include "Grid_Hierarchy_Projection.h"
#include "Inner_Product_Helper.h"
#include "Multigrid_Solver.h"

namespace Nova{
template<class Base_struct_type,class Multigrid_struct_type,class T,int d>
class CG_System: public Krylov_System_Base<T>
{
    using Base                      = Krylov_System_Base<T>;
    using Vector_Base               = Krylov_Vector_Base<T>;
    using Channel_Vector            = Vector<T Base_struct_type::*,d>;
    using Hierarchy                 = Grid_Hierarchy<Base_struct_type,T,d>;
    using Hierarchy_Projection      = Grid_Hierarchy_Projection<Base_struct_type,T,d>;

  public:
    Hierarchy& hierarchy;
    Channel_Vector& gradient_channels;
    const int mg_levels;
    Multigrid_Solver<Base_struct_type,Multigrid_struct_type,T,d> multigrid_solver;
    const int boundary_smoothing_iterations,interior_smoothing_iterations,bottom_smoothing_iterations;

    CG_System(Hierarchy& hierarchy_input,Channel_Vector& gradient_channels_input,const int mg_levels_input,
              const int boundary_smoothing_iterations_input,const int interior_smoothing_iterations_input,
              const int bottom_smoothing_iterations_input)
        :Base(true,false),hierarchy(hierarchy_input),gradient_channels(gradient_channels_input),mg_levels(mg_levels_input),
        multigrid_solver(hierarchy,mg_levels),boundary_smoothing_iterations(boundary_smoothing_iterations_input),
        interior_smoothing_iterations(interior_smoothing_iterations_input),bottom_smoothing_iterations(bottom_smoothing_iterations_input)
    {multigrid_solver.Initialize();}

    ~CG_System() {}

    void Set_Boundary_Conditions(Vector_Base& v) const {}
    void Project_Nullspace(Vector_Base& v) const {}

    void Multiply(const Vector_Base& v,Vector_Base& result) const
    {
        T Base_struct_type::* v_channel         = CG_Vector<Base_struct_type,T,d>::Cg_Vector(v).channel;
        T Base_struct_type::* result_channel    = CG_Vector<Base_struct_type,T,d>::Cg_Vector(result).channel;
        
        Hierarchy_Projection::Compute_Laplacian(hierarchy,gradient_channels,v_channel,result_channel);
    }

    void Project(Vector_Base& v) const
    {
        T Base_struct_type::* v_channel         = CG_Vector<Base_struct_type,T,d>::Cg_Vector(v).channel;

        for(int level=0;level<hierarchy.Levels();++level)
            Clear_Non_Active<Base_struct_type,T,d>(hierarchy.Allocator(level),hierarchy.Blocks(level),v_channel);
    }

    double Inner_Product(const Vector_Base& v1,const Vector_Base& v2) const
    {
        const Hierarchy& v1_hierarchy           = CG_Vector<Base_struct_type,T,d>::Hierarchy(v1);
        const Hierarchy& v2_hierarchy           = CG_Vector<Base_struct_type,T,d>::Hierarchy(v2);
        T Base_struct_type::* const v1_channel  = CG_Vector<Base_struct_type,T,d>::Cg_Vector(v1).channel;
        T Base_struct_type::* const v2_channel  = CG_Vector<Base_struct_type,T,d>::Cg_Vector(v2).channel;
        assert(&hierarchy == &v1_hierarchy);
        assert(&hierarchy == &v2_hierarchy);

        double result=0;

        for(int level=0;level<hierarchy.Levels();++level)
            Inner_Product_Helper<Base_struct_type,T,d>(hierarchy.Allocator(level),hierarchy.Blocks(level),v1_channel,
                                                  v2_channel,result,(unsigned)Cell_Type_Interior);

        return result;
    }

    T Convergence_Norm(const Vector_Base& v) const
    {
        T Base_struct_type::* v_channel         = CG_Vector<Base_struct_type,T,d>::Cg_Vector(v).channel;
        T max_value=(T)0.;

        for(int level=0;level<hierarchy.Levels();++level)
            Convergence_Norm_Helper<Base_struct_type,T,d>(hierarchy.Allocator(level),hierarchy.Blocks(level),
                                                     v_channel,max_value,(unsigned)Cell_Type_Interior);

        return max_value;
    }

    void Apply_Preconditioner(const Vector_Base& r,Vector_Base& z) const
    {
        T Base_struct_type::* r_channel         = CG_Vector<Base_struct_type,T,d>::Cg_Vector(r).channel;
        T Base_struct_type::* z_channel         = CG_Vector<Base_struct_type,T,d>::Cg_Vector(z).channel;

        multigrid_solver.Initialize_Right_Hand_Side(r_channel);
        multigrid_solver.Initialize_Guess();
        multigrid_solver.V_Cycle(boundary_smoothing_iterations,interior_smoothing_iterations,bottom_smoothing_iterations);

        // clear z
        for(int level=0;level<hierarchy.Levels();++level)
            SPGrid::Clear<Base_struct_type,T,d>(hierarchy.Allocator(level),hierarchy.Blocks(level),z_channel);

        // copy u from multigrid hierarchy
        multigrid_solver.Copy_Channel_Values(z_channel,multigrid_solver.u_channel,false);
    }
};
}
#endif
