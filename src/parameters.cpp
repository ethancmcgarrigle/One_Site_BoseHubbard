#include "parameters.h"
#include "global.h"

JSONParser::JSONParser(std::string const &filename)
  : parserbase(filename)
{
  FILE *fp = fopen(filename.c_str(), "r");
  if(fp==nullptr)
  {
    std::cerr << "JSONParser: Error opening file " << filename << std::endl;
    return;
  }

  char readBuffer[65536];
  rapidjson::FileReadStream instream(fp, readBuffer, sizeof(readBuffer));
  _docjson.ParseStream(instream);
  fclose(fp);

  if(!_docjson.IsObject())
  {
    std::cerr << "JSONParser: Not a JSON object" << std::endl;
    return;
  }

  _initialized = true;
}


bool JSONParser::getDouble(std::string key, std::list<std::string> block, double &value) const
{
  if(!_initialized)
    return false;

  // Create the composite path string
  std::string keypath;
  for(auto b : block)
    keypath += "/"+b;
  // Get a json subdocument for the specified path
  rapidjson::Value const *jsonval = rapidjson::Pointer(keypath.c_str()).Get(_docjson);
  // Validate
  if(!jsonval)
  {
    std::cout << "JSONParser: Invalid JSON path " << keypath << "\n";
    return false;
  }
  if(!jsonval->IsObject())
  {
    std::cout << "JSONParser: Invalid JSON path " << keypath << "\n";
    return false;
  }
  if(!jsonval->HasMember(key.c_str()))
  {
    std::cerr << "JSONParser: key " << key << " not found in path " << keypath << "\n";
    return false;
  }
  if(!(*jsonval)[key.c_str()].IsNumber())
  {
    std::cerr << "JSONParser: key " << key << " in path " << keypath << " is not a number\n";
    return false;
  }
  // Retrieve value
  value = (*jsonval)[key.c_str()].GetDouble();

  return true;
}

bool JSONParser::getInt(std::string key, std::list<std::string> block, int &value) const
{
  if(!_initialized)
    return false;

  // Create the composite path string
  std::string keypath;
  for(auto b : block)
    keypath += "/"+b;
  // Get a json subdocument for the specified path
  rapidjson::Value const *jsonval = rapidjson::Pointer(keypath.c_str()).Get(_docjson);
  // Validate
  if(!jsonval)
  {
    std::cout << "JSONParser: Invalid JSON path " << keypath << "\n";
    return false;
  }
  if(!jsonval->IsObject())
  {
    std::cout << "JSONParser: Invalid JSON path " << keypath << "\n";
    return false;
  }
  if(!jsonval->HasMember(key.c_str()))
  {
    std::cerr << "JSONParser: key " << key << " not found in path " << keypath << "\n";
    return false;
  }
  if(!(*jsonval)[key.c_str()].IsNumber())
  {
    std::cerr << "JSONParser: key " << key << " in path " << keypath << " is not a number\n";
    return false;
  }
  // Retrieve value
  value = (*jsonval)[key.c_str()].GetInt();

  return true;
}

bool JSONParser::getString(std::string key, std::list<std::string> block, std::string &value) const
{
  if(!_initialized)
    return false;

  // Create the composite path string
  std::string keypath;
  for(auto b : block)
    keypath += "/"+b;
  // Get a json subdocument for the specified path
  rapidjson::Value const *jsonval = rapidjson::Pointer(keypath.c_str()).Get(_docjson);
  // Validate
  if(!jsonval)
  {
    std::cout << "JSONParser: Invalid JSON path " << keypath << "\n";
    return false;
  }
  if(!jsonval->IsObject())
  {
    std::cout << "JSONParser: Invalid JSON path " << keypath << "\n";
    return false;
  }
  if(!jsonval->HasMember(key.c_str()))
  {
    std::cerr << "JSONParser: key " << key << " not found in path " << keypath << "\n";
    return false;
  }
  if(!(*jsonval)[key.c_str()].IsString())
  {
    std::cerr << "JSONParser: key " << key << " in path " << keypath << " is not a string\n";
    return false;
  }
  // Retrieve value
  value = (*jsonval)[key.c_str()].GetString();

  return true;
}

bool JSONParser::getBool(std::string key, std::list<std::string> block, bool &value) const
{
  if(!_initialized)
    return false;

  // Create the composite path string
  std::string keypath;
  for(auto b : block)
    keypath += "/"+b;
  // Get a json subdocument for the specified path
  rapidjson::Value const *jsonval = rapidjson::Pointer(keypath.c_str()).Get(_docjson);
  // Validate
  if(!jsonval)
  {
    std::cout << "JSONParser: Invalid JSON path " << keypath << "\n";
    return false;
  }
  if(!jsonval->IsObject())
  {
    std::cout << "JSONParser: Invalid JSON path " << keypath << "\n";
    return false;
  }
  if(!jsonval->HasMember(key.c_str()))
  {
    std::cerr << "JSONParser: key " << key << " not found in path " << keypath << "\n";
    return false;
  }
  if(!(*jsonval)[key.c_str()].IsBool())
  {
    std::cerr << "JSONParser: key " << key << " in path " << keypath << " is not a string\n";
    return false;
  }
  // Retrieve value
  value = (*jsonval)[key.c_str()].GetBool();
  return true;
}

template<typename T>
bool JSONParser::getVector(std::string key, std::list<std::string> block, std::vector<T> &value) const
{
  codeerror_abort("Parser function not yet added",__FILE__,__LINE__);
  return false;
}

bool JSONParser::getComplex(std::string key, std::list<std::string> block, std::complex<double> &value) const
{
  codeerror_abort("Parser function not yet added",__FILE__,__LINE__);
  return false;
}



//----------------------------------------------




YAMLParser::YAMLParser(std::string const &filename)
  : parserbase(filename)
  , _ymlconfig(nullptr)
{
  YAML::Node config = YAML::LoadFile(filename);
  _ymlconfig = new YAML::Node(config);
}

YAMLParser::~YAMLParser()
{
  delete _ymlconfig;
}

bool YAMLParser::getDouble(std::string key, std::list<std::string> block, double &value) const
{
  // We must clone the YML nodes, otherwise manipulating the node variable (e.g., to descend into subkeys)
  // will modify the datastructure (e.g., creating self-references)
  YAML::Node node = YAML::Clone(*_ymlconfig);
  for(auto b : block) // Descend into block
  {
    if(!node[b])
    {
      std::cerr << "YAMLParser: block " << b << " not found in document.\n";
      return false;
    }
    node = YAML::Clone(node[b]);
  }
  if(node[key])
  {
    value = node[key].as<double>();
    return true;
  }
  else
  {
    std::cerr << "YAMLParser: key " << key << " not found in document.\n";
    return false;
  }
}

bool YAMLParser::getInt(std::string key, std::list<std::string> block, int &value) const
{
  // We must clone the YML nodes, otherwise manipulating the node variable (e.g., to descend into subkeys)
  // will modify the datastructure (e.g., creating self-references)
  YAML::Node node = YAML::Clone(*_ymlconfig);
  for(auto b : block) // Descend into block
  {
    if(!node[b])
    {
      std::cerr << "YAMLParser: block " << b << " not found in document.\n";
      return false;
    }
    node = YAML::Clone(node[b]);
  }
  if(node[key])
  {
    value = node[key].as<int>();
    return true;
  }
  else
  {
    std::cerr << "YAMLParser: key " << key << " not found in document.\n";
    return false;
  }
}

bool YAMLParser::getString(std::string key, std::list<std::string> block, std::string &value) const
{
  // We must clone the YML nodes, otherwise manipulating the node variable (e.g., to descend into subkeys)
  // will modify the datastructure (e.g., creating self-references)
  YAML::Node node = YAML::Clone(*_ymlconfig);
  for(auto b : block) // Descend into block
  {
    if(!node[b])
    {
      std::cerr << "YAMLParser: block " << b << " not found in document.\n";
      return false;
    }
    node = YAML::Clone(node[b]);
  }
  if(node[key])
  {
    value = node[key].as<std::string>();
    return true;
  }
  else
  {
    std::cerr << "YAMLParser: key " << key << " not found in document.\n";
    return false;
  }
}

bool YAMLParser::getBool(std::string key, std::list<std::string> block, bool &value) const
{
  // We must clone the YML nodes, otherwise manipulating the node variable (e.g., to descend into subkeys)
  // will modify the datastructure (e.g., creating self-references)
  YAML::Node node = YAML::Clone(*_ymlconfig);
  for(auto b : block) // Descend into block
  {
    if(!node[b])
    {
      std::cerr << "YAMLParser: block " << b << " not found in document.\n";
      return false;
    }
    node = YAML::Clone(node[b]);
  }
  if(node[key])
  {
    value = node[key].as<bool>();
    return true;
  }
  else
  {
    std::cerr << "YAMLParser: key " << key << " not found in document.\n";
    return false;
  }
}

template<typename T>
bool YAMLParser::getVector(std::string key, std::list<std::string> block, std::vector<T> &value) const
{
  // We must clone the YML nodes, otherwise manipulating the node variable (e.g., to descend into subkeys)
  // will modify the datastructure (e.g., creating self-references)
  YAML::Node node = YAML::Clone(*_ymlconfig);
  for(auto b : block) // Descend into block
  {
    if(!node[b])
    {
      std::cerr << "YAMLParser: block " << b << " not found in document.\n";
      return false;
    }
    node = YAML::Clone(node[b]);
  }
  if(YAML::Node param = node[key])
  {
    assert(param.IsSequence());
    if(param.size() == value.size())
    {
      value = param.as<std::vector<T>>();
      return true;
    }
    else
      return false;
  }
  else
  {
    std::cerr << "YAMLParser: key " << key << " not found in document.\n";
    return false;
  }
}

bool YAMLParser::getComplex(std::string key, std::list<std::string> block, std::complex<double> &value) const
{
  // We must clone the YML nodes, otherwise manipulating the node variable (e.g., to descend into subkeys)
  // will modify the datastructure (e.g., creating self-references)
  std::vector<double> tmp(2);
  YAML::Node node = YAML::Clone(*_ymlconfig);
  for(auto b : block) // Descend into block
  {
    if(!node[b])
    {
      std::cerr << "YAMLParser: block " << b << " not found in document.\n";
      return false;
    }
    node = YAML::Clone(node[b]);
  }
  if(YAML::Node param = node[key])
  {
    assert(param.IsSequence());
    // Parse complex as a two-component vector
    tmp = param.as<std::vector<double>>();
    if(param.size() != 2)
      usererror_exit(std::string("YAMLParser: key "+key+" is not two-component complex").c_str(),__FILE__,__LINE__);
    value.real(tmp[0]);
    value.imag(tmp[1]);
    return true;
  }
  else
  {
    std::cerr << "YAMLParser: key " << key << " not found in document.\n";
    return false;
  }
}

//----------------------------------------------




TOMLParser::TOMLParser(std::string const &filename)
  : parserbase(filename)
  , _tomltable(nullptr)
{
  try {
    _tomltable = cpptoml::parse_file(filename);
  } catch (...) {
    std::cerr << "TOMLParser: Error opening file " << filename << std::endl;
  }

  _initialized = true;
}

bool TOMLParser::getDouble(std::string key, std::list<std::string> block, double &value) const
{
  std::string keypath;
  for(auto b : block)
    keypath += b+".";
  keypath += key;
  auto pval = _tomltable->get_qualified_as<double>(keypath);
  if(pval)
  {
    value = *pval;
    return true;
  } else {
    std::cerr << "TOMLParser: key " << keypath << " is not a floating-point number" << std::endl;
    return false;
  }
}

bool TOMLParser::getInt(std::string key, std::list<std::string> block, int &value) const
{
  std::string keypath;
  for(auto b : block)
    keypath += b+".";
  keypath += key;
  auto pval = _tomltable->get_qualified_as<int>(keypath);
  if(pval)
  {
    value = *pval;
    return true;
  } else {
    std::cerr << "TOMLParser: key " << keypath << " is not an integer" << std::endl;
    return false;
  }
}

bool TOMLParser::getString(std::string key, std::list<std::string> block, std::string &value) const
{
  std::string keypath;
  for(auto b : block)
    keypath += b+".";
  keypath += key;
  auto pval = _tomltable->get_qualified_as<std::string>(keypath);
  if(pval)
  {
    value = *pval;
    return true;
  } else {
    std::cerr << "TOMLParser: key " << keypath << " is not an integer" << std::endl;
    return false;
  }
}

bool TOMLParser::getBool(std::string key, std::list<std::string> block, bool &value) const
{
  std::string keypath;
  for(auto b : block)
    keypath += b+".";
  keypath += key;
  auto pval = _tomltable->get_qualified_as<bool>(keypath);
  if(pval)
  {
    value = *pval;
    return true;
  } else {
    std::cerr << "TOMLParser: key " << keypath << " is not an integer" << std::endl;
    return false;
  }
}

template<typename T>
bool TOMLParser::getVector(std::string key, std::list<std::string> block, std::vector<T> &value) const
{
  codeerror_abort("TOMLParser: getVector function not yet added",__FILE__,__LINE__);
  return false;
}

bool TOMLParser::getComplex(std::string key, std::list<std::string> block, std::complex<double> &value) const
{
  codeerror_abort("TOMLParser: getComplex function not yet added",__FILE__,__LINE__);
  return false;
}


// Template specialization stubs
template bool YAMLParser::getVector(std::string, std::list<std::string>, std::vector<int> &) const;
template bool YAMLParser::getVector(std::string, std::list<std::string>, std::vector<size_t> &) const;
template bool YAMLParser::getVector(std::string, std::list<std::string>, std::vector<double> &) const;
