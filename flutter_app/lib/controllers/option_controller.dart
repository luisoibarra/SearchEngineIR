

import 'package:get/get.dart';
import 'package:ir_search_engine/services/api_configuration_service.dart';

class OptionController extends GetxController {
  // Services
  final _apiConfigurationService = Get.find<IApiConfigurationService>();

  @override
  void onInit() {
    super.onInit();
    _apiConfigurationService.getHost().then((value) => host.value = value);
    _apiConfigurationService.getPort().then((value) => port.value = value);
  }

  // Properties
  final host = "".obs;
  final port = 0.obs; 

  // Commands
  Future<void> saveChanges() async {
    await _apiConfigurationService.setHost(host.value);
    await _apiConfigurationService.setPort(port.value);
  }

}