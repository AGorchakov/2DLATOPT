/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   testuniformpp.cpp
 * Author: mposypkin
 *
 * Created on June 5, 2017, 3:42 PM
 */
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <limits>
#include <pointgen/randpointgen.hpp>
#include <common/vec.hpp>
//#include <pairpotentials.hpp>
#include <methods/rosenbrock/rosenbrockmethod.hpp>
#include <methods/advancedcoordescent/advancedcoordescent.hpp>
#include <funccnt.hpp>
#include "tsofproblem.hpp"

#include <sys/time.h>
double MyGetTime(){
struct timeval t0;
gettimeofday(&t0, NULL);
return t0.tv_sec + (t0.tv_usec / 1000000.0);
}


/*
 * 
 */

void acdSearch(double& v, double* x, const COMPI::MPProblem<double>& prob) {
    LOCSEARCH::AdvancedCoordinateDescent<double> desc(prob);
    desc.getOptions().mSearchType = LOCSEARCH::AdvancedCoordinateDescent<double>::SearchTypes::NO_DESCENT;
//    desc.getOptions().mDoTracing = true;
    
//    std::cout << "ADC before v = " << v << std::endl;
    bool rv = desc.search(x, v);
//    std::cout << "ADC after v = " << v << std::endl;
}


void rosenSearch(double& v, double* x, const COMPI::MPProblem<double>& prob) {
    LOCSEARCH::RosenbrockMethod<double> desc(prob);
    desc.getOptions().mHInit = std::vector<double>(prob.mBox->mDim, 1.);
    desc.getOptions().mMaxStepsNumber = 10000;
    desc.getOptions().mMinGrad = 1e-3;
    desc.getOptions().mHLB = 1e-4 * desc.getOptions().mMinGrad;
    
    desc.getOptions().mDoOrt = false;
    desc.getOptions().mDoTracing = true;
    
    std::cout << "Rosenbrock before v = " << v << std::endl;
    bool rv = desc.search(x, v);
    std::cout << "Rosenbrock after v = " << v << std::endl;
}

void search(double& v, double* x, const COMPI::MPProblem<double>& prob) {
    printf("n=\n");
    acdSearch(v, x, prob);
    //rosenSearch(v, x, prob);
}


int main(int argc, char** argv) {
    constexpr double length = 16;
    constexpr int nlayers = 4;
    constexpr int npoints = 32000;

    double ** frez = new double*[npoints];
    for(int i = 0; i < npoints; i++) frez[i] = new double[15];
    double start = MyGetTime();
    double rec;

#pragma omp parallel for schedule(dynamic), num_threads(40)
    for(int i = 0; i < npoints; i ++) {
       std::vector<lattice::AtomTypes> atoms(nlayers, lattice::AtomTypes::CARBON);
       lattice::TsofPotentialProblem uprob(length, atoms);
       //auto obj = std::make_shared<COMPI::FuncCnt<double>>(uprob.mObjectives.at(0));
//       auto fcnt = std::make_shared<COMPI::FuncCnt<double>>(uprob.mObjectives[0]);
       double v = std::numeric_limits<double>::max();
//       uprob.mObjectives.pop_back();
//       uprob.mObjectives.push_back(fcnt);
       snowgoose::RandomPointGenerator<double> rg(*(uprob.mBox), 1, i);
       const int n = uprob.mVarTypes.size();
       double x[n];
       //double bestx[n];
       rg.getPoint(x);
       double nv = uprob.mObjectives[0]->func(x);
       if(x[4] >= x[5]) {double tmp = x[4]; x[4] = x[5]; x[5] = tmp;}
       if(x[7] >= x[8]) {double tmp = x[7]; x[7] = x[8]; x[8] = tmp;}
       if(x[10] >= x[11]) {double tmp = x[10]; x[10] = x[11]; x[11] = tmp;}

       search(nv, x, uprob);
#pragma omp critical
{
       if (nv < rec) rec = nv;
}
       frez[i][0] =  MyGetTime()-start;
       frez[i][1] = rec;
       frez[i][2] = nv;
       for(int j = 0; j < 12; j++) frez[i][j+3] = x[j];
         
    }

    FILE * point = fopen("acd_32000.txt", "w");
    fprintf(point, "time, rec,func,x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11\n");
    for(int i = 0; i < npoints; i++) 
       if(1 || frez[i][1+4] < frez[i][1+5] &&
          frez[i][1+7] < frez[i][1+8] &&
          frez[i][1+10] < frez[i][1+11]) {
          for(int j = 0; j < 14; j++) fprintf(point, "%18.12lf,", frez[i][j]);
          fprintf(point, "%18.12lf\n", frez[i][14]);
    }
    fclose(point);

    return 0;
}

