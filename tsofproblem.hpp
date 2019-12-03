/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   tpproblem.hpp
 * Author: agorchakov
 *
 * Tersoff potential problem factory
 * 
 * Created on Jan 24, 2019, 13:28
 */

#ifndef UNIFORMPP_HPP
#define UNIFORMPP_HPP

#include <vector>
#include <mpproblem.hpp>
#include "atoms.hpp"
#include "latticedata.hpp"
#include "energyfunc.hpp"
//#include "pairpotentials.hpp"
#include "latticeutils.hpp"
#include "tsofenergy.hpp"
#include "carbontersoff.hpp"

namespace lattice {

    /**
     * The fragment of a lattice of uniform atoms interacting with the pair potential
     */
    class TsofPotentialProblem : public COMPI::MPProblem<double> {
    public:

        static void stdBoxFill(const LatticeData& latticeData, snowgoose::Box<double>& box) {
            for (int i = 0; i < latticeData.mNumLayers; i++) {
                const int j = LatticeData::ParamInd::RecSize * i;
                // height
                box.mA[j + LatticeData::ParamInd::Height] = 0.5;
                box.mB[j + LatticeData::ParamInd::Height] = 4;
                // displacement
                if (i == 0) {
                    box.mA[j + LatticeData::ParamInd::Displ] = 0;
                    box.mB[j + LatticeData::ParamInd::Displ] = 0;
                } else {
                    box.mA[j + LatticeData::ParamInd::Displ] = 0;
                    box.mB[j + LatticeData::ParamInd::Displ] = 4;

                }
                // stride
                box.mA[j + LatticeData::ParamInd::Stride] = 0.4;
                box.mB[j + LatticeData::ParamInd::Stride] = 4;
            }
        }

        /**
         * Constructor 
         * @param potent pair potential 
         * @param length the length of the material
         * @param atomTypes atom types (one per layer)
         */
        TsofPotentialProblem(double length, 
                             const std::vector<lattice::AtomTypes>& atomTypes, 
                             std::function<void(const LatticeData& , snowgoose::Box<double>&) > boxFill = stdBoxFill) {
            const int nlayers = atomTypes.size();
            /*** Setup lattice data */
            lattice::LatticeData data(nlayers);
            data.mRadius = lattice::stdCutoff;
            data.mLayersAtoms = atomTypes;

            /*** Setup potential */
            lattice::TersoffParams tparam;
            lattice::fillCarbonParametersTersoffOriginal(tparam);
            lattice::TersoffUtils tutils(tparam);
            lattice::LatticeUtils lut(data);
            auto enrg = std::make_shared<TersoffEnergy>(lut, tutils, length);
            mObjectives.push_back(std::make_shared<EnergyFunc>(enrg));
            const int n = data.mNumLayers * LatticeData::ParamInd::RecSize;
            mVarTypes.assign(n, COMPI::MPProblem<double>::VariableTypes::GENERIC);
            mBox = new snowgoose::Box<double>(n);
            boxFill(data, *mBox);
        }

    private:

    };

}


#endif /* UNIFORMPP_HPP */

