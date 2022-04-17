

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:ir_search_engine/models/document_model.dart';

class SearchItemWidget extends StatelessWidget {

  final DocumentModel document;

  const SearchItemWidget({Key? key, required this.document}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Text(document.documentName),
    );
  }

  
}