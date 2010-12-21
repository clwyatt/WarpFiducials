#include <cstdlib>

#include <iostream>
using std::cout; using std::endl;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include <fstream>
using std::ifstream; using std::ofstream;

#include <sstream>
using std::istringstream;

#include <itkImage.h>

struct Fiducial
{
  string name;
  float x, y, z;
  bool active, visible;
};

typedef vector<Fiducial> FiducialType;

bool readfid(const string & inputfidfile, FiducialType & infid);

int main(int argc, char *argv[])
{
  // command line options
  // warpfids inputfield.mha inputfids.fscv outputfids.fcsv
  if(argc != 5)
    {
    cout << "Error: missing option.\n" << endl;
    cout << "Usage: " << argv[0] << " target source inputfids.fscv outputfids.fcsv\n" << endl;
    return EXIT_FAILURE;
    }
  string targetfile = argv[1];
  string sourcefile = argv[1];
  string inputfidfile = argv[3];
  string outputfidfile = argv[4];

  // read and register source and target

  // read fiducials csv file
  FiducialType infid;
  if( readfid(inputfidfile, infid) )
    {
    cout << "Error reading input fiducial file. Halting." << endl;
    return EXIT_FAILURE;
    }

  // warp fiducials
  FiducialType outfid = infid;

  // save warped fiducials csv file

  for(unsigned int i = 0; i < outfid.size(); ++i)
    {
    Fiducial p = outfid[i];
    cout << p.name.c_str() << ","
	 << p.x << "," << p.y << "," << p.z << ","
	 << p.active << "," << p.visible << endl;
    }

  return EXIT_SUCCESS;
}

bool readfid(const string & inputfidfile, FiducialType & infid)
{
  bool err = false;
  ifstream infidfile( inputfidfile.c_str() );

  if(infidfile.fail() )
    {
    cout << "Could not open file: " << inputfidfile.c_str() << endl;
    return true;
    }

  while(!infidfile.eof() && infidfile.good() )
    {
    string line;
    getline(infidfile, line);

    if( line.empty() || (line[0] == '#') ) continue;

    istringstream iss(line);
    Fiducial p;
    getline(iss, p.name, ',');

    string field;
    getline(iss, field, ',');
    istringstream xstream(field);
    if(!(xstream >> p.x) )
      {
      cout << "Warning: error reading x field on line:\n" << line << endl;
      err = true;
      continue;
      }

    getline(iss, field, ',');
    istringstream ystream(field);
    if(!(ystream >> p.y) )
      {
      cout << "Warning: error reading y field on line:\n" << line << endl;
      err = true;
      continue;
      }

    getline(iss, field, ',');
    istringstream zstream(field);
    if(!(zstream >> p.z) )
      {
      cout << "Warning: error reading z field on line:\n" << line << endl;
      err = true;
      continue;
      }

    getline(iss, field, ',');
    istringstream astream(field);
    if(!(astream >> p.active) )
      {
      cout << "Warning: error reading active flag on line:\n" << line << endl;
      err = true;
      continue;
      }

    getline(iss, field, ',');
    istringstream vstream(field);
    if(!(vstream >> p.visible) )
      {
      cout << "Warning: error reading visible flag on line:\n" << line << endl;
      err = true;
      continue;
      }

    infid.push_back(p);
    }
  infidfile.close();

  return err;
}
