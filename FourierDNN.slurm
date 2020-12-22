#!/bin/bash                                                     
#SBATCH --job-name=fourierdnn2lxa                 
#SBATCH --partition=gpu                             
#SBATCH -n 10                                           
#SBATCH --cpus-per-task=4                         
#SBATCH --gres=gpu:1                                
#SBATCH --output=%j.out                            
#SBATCH --error=%j.err                                
#SBATCH --time=24:00:10                             
#SBATCH --mail-type=all
#SBATCH --mail-user=lixian9131@163.com   
#module add anaconda3/2020.02          
#python cs_Laplace5d.py