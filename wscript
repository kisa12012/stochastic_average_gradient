#!/usr/bin/python

VERSION = '0.0.1'
APPNAME = 'linearly_convergence_sgd'

srcdir = '.'
blddir = 'build'

def options(opt):
  opt.load('compiler_cxx')

def configure(conf):
  conf.load('compiler_cxx')
  conf.check_cfg(package = 'eigen3')
  conf.env.append_value('CXXFLAGS', ['-W', '-Wall', '-std=c++0x'])

def build(bld):
  bld.recurse('src')
