

import 'package:get/get.dart';
import 'package:ir_search_engine/services/api_configuration_service.dart';
import 'package:ir_search_engine/services/retrieve_documents_service.dart';

abstract class IDependencyInjection {
  Future<void> putServices();
}

class DependencyInjection extends GetxService implements IDependencyInjection {
  
  @override
  Future<void> putServices() async {
    Get.put<IApiConfigurationService>(ApiConfigurationService());
    // Get.lazyPut<IRetrieveDocumentsService>(() => RetrieveDocumentsService());
    Get.put<IRetrieveDocumentsService>(MockRetrieveDocumentsService());
  }

}