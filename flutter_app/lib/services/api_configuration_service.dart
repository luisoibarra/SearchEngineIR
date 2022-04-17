import 'package:get/get.dart';
import 'package:shared_preferences/shared_preferences.dart';

abstract class IApiConfigurationService {

  Future<String> getHost();
  Future<int> getPort();
  Future<void> setHost(String host);
  Future<void> setPort(int port);
  Future<String> getUrl(String path, {Map<String, String>? queryParams});

}

class ApiConfigurationService extends GetxService implements IApiConfigurationService {
  static const _hostKey = "host_key";
  static const _hostDefault = "192.168.43.143";

  static const _portKey = "port_key";
  static const _portDefault = "5555";

  Future<String> _getAndSaveKey(String key, String defaultValue) async {
    final sharedPreferences = await SharedPreferences.getInstance();
    final value = sharedPreferences.getString(key);
    if (value == null) {
      await sharedPreferences.setString(key, defaultValue);
    }
    return value ?? defaultValue;
  }

  @override
  Future<String> getHost() {
    final host = _getAndSaveKey(_hostKey, _hostDefault);
    return host;
  }

  @override
  Future<int> getPort() async {
    final port = await _getAndSaveKey(_portKey, _portDefault);
    return int.parse(port);
  }

  @override
  Future<void> setHost(String host) async {
    final sharedPreferences = await SharedPreferences.getInstance();
    await sharedPreferences.setString(_hostKey, host);
  }

  @override
  Future<void> setPort(int port) async {
    final sharedPreferences = await SharedPreferences.getInstance();
    sharedPreferences.setString(_hostKey, port.toString());
  }

  @override
  Future<String> getUrl(String path, {Map<String, String>? queryParams}) async {
    final host = await getHost();
    final port = await getPort();
    return Uri(
        host: host, port: port, path: path, queryParameters: queryParams).toString();
  }
}
