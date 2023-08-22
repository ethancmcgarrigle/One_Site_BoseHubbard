#ifndef _PARAMETERS_H_
#define _PARAMETERS_H_

#include "../include/rapidjson/filereadstream.h"
#include "../include/rapidjson/document.h"
#include "../include/rapidjson/pointer.h"
// femtoyaml works well, but it requires C++17 - perhaps too restrictive.
//#include "../include/femtoyaml/femtoyaml.hpp"
#include "../include/yaml-cpp/include/yaml-cpp/yaml.h"
#include "../include/cpptoml/cpptoml.h"
#include <complex>

class parserbase {
  public:
    explicit parserbase(std::string const &filename)
      : _filename(filename)
      , _initialized(false)
    {};
    virtual ~parserbase() {};

    virtual bool getDouble(std::string key, std::list<std::string> block, double &value) const = 0; /// Return false if fail for any reason
    virtual bool getInt(std::string key, std::list<std::string> block, int &value) const = 0; /// Return false if fail for any reason
    virtual bool getString(std::string key, std::list<std::string> block, std::string &value) const = 0; /// Return false if fail for any reason
    virtual bool getBool(std::string key, std::list<std::string> block, bool &value) const = 0; /// Return false if fail for any reason
    virtual bool getComplex(std::string key, std::list<std::string> block, std::complex<double> &value) const = 0; /// Return false if fail for any reason

  protected:
    std::string const _filename;
    bool _initialized;
};

// Three parser options.
// JSON is widely supported but the format is not as human readable.
// YAML is human readable. v1.2 is a superset of JSON. YAML retains much of the flexibility of JSON.
// TOML is the simplest markup format, but the nesting of sections is a little cumersome.
class JSONParser : public parserbase {
  public:
    // ctor
    explicit JSONParser(std::string const &filename);
    // dtor
    ~JSONParser() {};

    virtual bool getDouble(std::string key, std::list<std::string> block, double &value) const; /// Return false if fail for any reason
    virtual bool getInt(std::string key, std::list<std::string> block, int &value) const; /// Return false if fail for any reason
    virtual bool getString(std::string key, std::list<std::string> block, std::string &value) const; /// Return false if fail for any reason
    virtual bool getBool(std::string key, std::list<std::string> block, bool &value) const; /// Return false if fail for any reason
    template<typename T> bool getVector(std::string key, std::list<std::string> block, std::vector<T> &value) const; /// Return false if fail for any reason
    virtual bool getComplex(std::string key, std::list<std::string> block, std::complex<double> &value) const; /// Return false if fail for any reason

  private:
    rapidjson::Document _docjson;
};

class YAMLParser : public parserbase {
  public:
    // ctor
    explicit YAMLParser(std::string const &filename);
    // dtor
    ~YAMLParser();

    virtual bool getDouble(std::string key, std::list<std::string> block, double &value) const; /// Return false if fail for any reason
    virtual bool getInt(std::string key, std::list<std::string> block, int &value) const; /// Return false if fail for any reason
    virtual bool getString(std::string key, std::list<std::string> block, std::string &value) const; /// Return false if fail for any reason
    virtual bool getBool(std::string key, std::list<std::string> block, bool &value) const; /// Return false if fail for any reason
    template<typename T> bool getVector(std::string key, std::list<std::string> block, std::vector<T> &value) const; /// Return false if fail for any reason
    virtual bool getComplex(std::string key, std::list<std::string> block, std::complex<double> &value) const; /// Return false if fail for any reason

  private:
    //femtoyaml::value const &_yml;
    YAML::Node *_ymlconfig;
};

class TOMLParser : public parserbase {
  public:
    // ctor
    explicit TOMLParser(std::string const &filename);
    // dtor
    ~TOMLParser() {};

    virtual bool getDouble(std::string key, std::list<std::string> block, double &value) const; /// Return false if fail for any reason
    virtual bool getInt(std::string key, std::list<std::string> block, int &value) const; /// Return false if fail for any reason
    virtual bool getString(std::string key, std::list<std::string> block, std::string &value) const; /// Return false if fail for any reason
    virtual bool getBool(std::string key, std::list<std::string> block, bool &value) const; /// Return false if fail for any reason
    template<typename T> bool getVector(std::string key, std::list<std::string> block, std::vector<T> &value) const; /// Return false if fail for any reason
    virtual bool getComplex(std::string key, std::list<std::string> block, std::complex<double> &value) const; /// Return false if fail for any reason

  private:
    std::shared_ptr<cpptoml::table> _tomltable;
};

#endif // _PARAMETERS_H_
