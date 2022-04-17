

import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:ir_search_engine/models/query_response_model.dart';
import 'package:ir_search_engine/services/retrieve_documents_service.dart';

class RetrievalController extends GetxController {
  // Services
  final _retrieveDocumentService = Get.find<IRetrieveDocumentsService>();

  // Properties
  late final TextEditingController query;
  final Rx<QueryResponseModel?> queryResponse = Rx<QueryResponseModel?>(null); 

  @override
  void onInit() {
    super.onInit();
    query = TextEditingController();
  }

  // Command
  Future<void> search() async {
    final response = await _retrieveDocumentService.getResults(query.text);
    queryResponse.value = response;
  }

}