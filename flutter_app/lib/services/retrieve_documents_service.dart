

import 'package:get/get.dart';
import 'package:ir_search_engine/models/document_model.dart';
import 'package:ir_search_engine/models/query_response_model.dart';
import 'package:ir_search_engine/services/api_configuration_service.dart';

abstract class IRetrieveDocumentsService {

  Future<QueryResponseModel?> getResults(String query);

}

class RetrieveDocumentsService extends GetConnect implements IRetrieveDocumentsService {

  final _apiConfigurationService = Get.find<ApiConfigurationService>();

  static const _getQueryPath = "query";

  @override
  Future<QueryResponseModel?> getResults(String query) async {
    final url = await _apiConfigurationService.getUrl(_getQueryPath);
    final response = await get(url, query: {"query": query}, decoder:(value) => QueryResponseModel.fromJson(value));
    return response.body;
  }

}

class MockRetrieveDocumentsService implements IRetrieveDocumentsService {
  @override
  Future<QueryResponseModel?> getResults(String query) async {
    return QueryResponseModel(documents: [
      DocumentModel(documentName: query + " 1"),
      DocumentModel(documentName: query + " 2"),
      DocumentModel(documentName: query + " 3"),
    ]);
  }

}