<p align="center">
  <a href="https://github.com/IgorBaratta/odd/"><img src="https://user-images.githubusercontent.com/15614155/74994585-39e82c80-542d-11ea-8cde-a2c2a6f95dbf.png" alt="odd- 1280x640" width="70%"/></a>
</p>

[![CircleCI](https://circleci.com/gh/IgorBaratta/odd.svg?style=shield&circle-token=665de032e391d8428935ec8940d183fbdd6576d7)](https://circleci.com/gh/IgorBaratta/odd)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage Status](https://coveralls.io/repos/github/IgorBaratta/odd/badge.svg?branch=master&t=RH1GCe)](https://coveralls.io/github/IgorBaratta/odd?branch=master)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

A simple distributed (MPI) domain decomposition library in python.

## Installation
`odd` requires `numpy`, `scipy` and `mpi4py`.

On OS X & Linux:

```sh
git clone git@github.com:IgorBaratta/odd.git
python3 -m pip odd/
```

or:

```
 pip3 install git+https://github.com/IgorBaratta/odd.git --upgrade
``` 

For a full experience we recommend to use `odd` in combination with the FEniCS libraries.
Please refer to the Demos for more examples and usage.

## Usage example

A few motivating and useful examples of how your product can be used. Spice this up with code blocks and potentially more screenshots.

_For more examples and usage, please refer to the [Wiki][wiki]._

## Development setup

Describe how to install all development dependencies and how to run an automated test-suite of some kind. Potentially do this for multiple platforms.

```sh
make install
npm test
```

## Meta

Igor A. Baratta – [@IgorBaratta](https://github.com/IgorBaratta) – baratta@ufmg.br

Distributed under the LGPL 3.0 license. See ``LICENSE`` for more information.

## Contributing

1. Fork it (<https://github.com/IgorBaratta/odd/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request
