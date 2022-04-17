import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:ir_search_engine/controllers/retrieval_controller.dart';
import 'package:ir_search_engine/views/option_view.dart';
import 'package:ir_search_engine/widgets/search_item_widget.dart';

class RetrievalView extends StatelessWidget {
  const RetrievalView({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Search Engine")),
      drawer: Drawer(child: ListView(
        padding: EdgeInsets.zero,
        children: const [
          DrawerHeader(child: Text("Options")),
          OptionView()
        ],
      )),
      body: GetX<RetrievalController>(
          init: RetrievalController(),
          builder: (controller) {
            return Container(
              margin: const EdgeInsets.symmetric(horizontal: 10),
              child: Column(
                children: [
                  Row(
                    children: [
                      Flexible(
                        child: TextFormField(
                            controller: controller.query,
                            decoration:
                                const InputDecoration(hintText: "Search")),
                      ),
                      IconButton(
                          onPressed: () => controller.search(),
                          icon: const Icon(Icons.search))
                    ],
                  ),
                  Flexible(
                    child: ListView.builder(
                        shrinkWrap: true,
                        itemCount:
                            controller.queryResponse.value?.documents.length ??
                                0,
                        itemBuilder: (context, index) {
                          final isNull = controller.queryResponse.value == null;
                          final item = !isNull &&
                                  controller.queryResponse.value!.documents
                                          .length >
                                      index
                              ? controller.queryResponse.value?.documents[index]
                              : null;
                          if (item != null) {
                            return SearchItemWidget(document: item);
                          }
                          return const Text("No More Results");
                        }),
                  )
                ],
              ),
            );
          }),
    );
  }
}
